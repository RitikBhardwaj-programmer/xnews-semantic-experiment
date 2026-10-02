"""
Stage 2 of V4 approach 1 (offline): does a maximum event lifetime and/or a
nightly merge-only pass improve the shipped v2 matcher?

The three labelled days are replayed CONTINUOUSLY (events can span days,
unlike the stage 1 replays, which started each day empty) with the shipped
v2 setup: variant B without temporal_score, TF-IDF vocabulary fitted on the
other days, consistent training, 20-member text cap. Each article is decided
by the model trained on the other two days (leave-one-day-out).

- lifetime cap k: an event whose first activity is more than k days before
  the article stops taking new articles
- merge pass (at each day boundary and after the last article): for every
  event active in the last 72 h, smallest first, the 5 most similar other
  active events (centroid cosine) are scored; the smaller event merges into
  the best one if its members' mean v2 probability against that event
  reaches the model threshold. Merge-only: nothing is ever split.

Labels are per day, so scoring is within each day (pooled over days, both
truth and prediction prefixed by day). "Mixed events": predicted events that,
within one day, contain articles of two or more true events.

Criteria (fixed before the run):
- merge pass ships if pairwise F1 improves with a 95% cluster-bootstrap
  interval above 0, B-cubed F1 is not worse, and mixed events do not increase
- lifetime cap ships if it is not worse on B-cubed and pairwise F1 (interval
  of the difference includes or exceeds 0) and mixed events do not increase

Usage:
    python evaluate_stage2.py
"""

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from cluster_metrics import bootstrap_bcubed, bootstrap_pairwise, summary
from evaluate_centroid_matching import temporal_score, update_centroid
from evaluate_stage1 import (DAYS, RefitIndex, choose_threshold, fit, load, model_decide, oracle_rows,
                             refit_index, shipped_columns)
from event_features import new_vectorizers, tfidf_text
from replay import CANDIDATE_LIMIT, INACTIVITY_DAYS, Event, day_gap

MEMBER_TEXT_CAP = 20
MERGE_WINDOW = pd.Timedelta(hours=72)
MERGE_CANDIDATES = 5

CONFIGS = [
    # name, lifetime cap (days), merge pass
    ("v2 continuous", None, False),
    ("v2 + cap 2 days", 2, False),
    ("v2 + cap 3 days", 3, False),
    ("v2 + merge pass", None, True),
    ("v2 + cap 3 days + merge pass", 3, True),
]


# ------------------------------------------------------------
# DATA, INDEX AND MODELS
# ------------------------------------------------------------

def combined(frames):
    articles = pd.concat([f.assign(day=d) for d, f in enumerate(frames)], ignore_index=True)
    return articles


def combined_index(articles):
    """Each article (and its candidate members) is seen through a vocabulary fitted
    on the OTHER days' texts, as in the stage 1b consistent setup."""
    texts = [tfidf_text(s, d if isinstance(d, str) else None)
             for s, d in zip(articles["stripped"], articles["description"])]
    matrices = {}
    for day in range(len(DAYS)):
        other = articles[articles["day"] != day]
        titles, text_vectorizer = new_vectorizers()
        titles.fit(other["stripped"])
        text_vectorizer.fit([tfidf_text(s, d if isinstance(d, str) else None)
                             for s, d in zip(other["stripped"], other["description"])])
        matrices[day] = (titles.transform(articles["stripped"]), text_vectorizer.transform(texts))
    return RefitIndex(articles.assign(title=articles["stripped"]), matrices,
                      articles["day"].to_numpy(), MEMBER_TEXT_CAP)


def train_for(frames, test):
    columns = shipped_columns()
    train_days = [d for d in range(len(DAYS)) if d != test]
    rows = pd.concat([
        oracle_rows(frames[d], refit_index(frames, d, [c for c in train_days if c != d], MEMBER_TEXT_CAP, False))
        for d in train_days])
    model = fit(rows, columns, "lr")
    return model, float(choose_threshold(model, rows, columns, False))


# ------------------------------------------------------------
# CONTINUOUS REPLAY WITH CAP AND MERGE PASS
# ------------------------------------------------------------

def run(articles, index, models, cap_days, merge):
    columns = shipped_columns()
    decide_for = {d: model_decide(m, columns, t, index, False, False) for d, (m, t) in models.items()}
    order = articles.sort_values(["created_at", "id"]).index
    events, parent = [], []
    assigned = pd.Series(-1, index=articles.index)
    current_day = None

    for row in order:
        article = articles.loc[row]
        now, day, vector = article["created_at"], article["day"], article["vector"]

        if merge and current_day is not None and day != current_day:
            merge_pass(events, parent, articles, index, models[day], now)
        current_day = day

        open_events = [
            k for k, e in enumerate(events)
            if parent[k] == k
            and e.last_activity >= now - pd.Timedelta(days=INACTIVITY_DAYS)
            and (cap_days is None or e.first_activity >= now - pd.Timedelta(days=cap_days))
        ]
        candidates = []
        if open_events:
            sims = np.stack([events[k].centroid for k in open_events]) @ vector
            for j in np.argsort(-sims)[:CANDIDATE_LIMIT]:
                k = open_events[j]
                gap = day_gap(article["published_at"], events[k].last_activity)
                candidates.append({"index": k, "similarity": float(sims[j]),
                                   "temporal_score": 0.5 if gap is None else float(temporal_score(np.array(gap))),
                                   "members": events[k].members})

        chosen = decide_for[day](article, candidates)
        if chosen is None:
            events.append(Event(centroid=update_centroid(None, 0, vector), first_activity=now, last_activity=now))
            parent.append(len(events) - 1)
            chosen = len(events) - 1
        else:
            event = events[chosen]
            event.centroid = update_centroid(event.centroid, len(event.members), vector)
        add_member(events[chosen], row, article)
        assigned[row] = chosen

    if merge:
        last = articles["day"].max()
        merge_pass(events, parent, articles, index, models[last], articles["created_at"].max() + pd.Timedelta(minutes=1))

    return assigned.map(lambda k: root_of(parent, k)).to_numpy()


def add_member(event, row, article):
    event.members.append(row)
    published = article["published_at"]
    if pd.notna(published):
        if len(event.members) == 1:
            event.first_activity = event.last_activity = published
        else:
            event.first_activity = min(event.first_activity, published)
            event.last_activity = max(event.last_activity, published)


def root_of(parent, k):
    while parent[k] != k:
        k = parent[k]
    return k


def merge_pass(events, parent, articles, index, model_threshold, now):
    model, threshold = model_threshold
    columns = shipped_columns()
    active = [k for k, e in enumerate(events)
              if parent[k] == k and e.last_activity >= now - MERGE_WINDOW]
    for a in sorted(active, key=lambda k: len(events[k].members)):
        if parent[a] != a:
            continue
        others = [k for k in active if k != a and parent[k] == k]
        if not others:
            continue
        sims = np.stack([events[k].centroid for k in others]) @ events[a].centroid
        best, best_p = None, -1.0
        for j in np.argsort(-sims)[:MERGE_CANDIDATES]:
            b = others[j]
            if len(events[b].members) < len(events[a].members):
                continue  # the smaller event merges into the larger one
            rows = []
            for m in events[a].members:
                vector = articles.at[m, "vector"]
                gap = day_gap(articles.at[m, "published_at"], events[b].last_activity)
                rows.append(index.features(m, events[b].members, float(events[b].centroid @ vector),
                                           0.5 if gap is None else float(temporal_score(np.array(gap)))))
            p = float(model.predict_proba(pd.DataFrame(rows)[columns])[:, 1].mean())
            if p > best_p:
                best, best_p = b, p
        if best is not None and best_p >= threshold:
            merge_into(events, parent, articles, a, best)


def merge_into(events, parent, articles, a, b):
    source, target = events[a], events[b]
    for m in source.members:
        target.centroid = update_centroid(target.centroid, len(target.members), articles.at[m, "vector"])
        target.members.append(m)
    target.first_activity = min(target.first_activity, source.first_activity)
    target.last_activity = max(target.last_activity, source.last_activity)
    parent[a] = b


# ------------------------------------------------------------
# SCORING
# ------------------------------------------------------------

def mixed_events(truth, pred):
    frame = pd.DataFrame({"t": truth, "p": pred})
    return int((frame.groupby("p")["t"].nunique() >= 2).sum())


def main():
    started = time.time()
    frames = [load(d) for d in DAYS]
    articles = combined(frames)
    index = combined_index(articles)
    models = dict(zip(range(len(DAYS)), Parallel(n_jobs=len(DAYS))(
        delayed(train_for)(frames, d) for d in range(len(DAYS)))))
    print(f"models: thresholds {[round(t, 2) for _, t in models.values()]} ({time.time() - started:.0f}s)", flush=True)

    outputs = Parallel(n_jobs=len(CONFIGS))(
        delayed(run)(articles, index, models, cap, merge) for _, cap, merge in CONFIGS)
    print(f"replays done ({time.time() - started:.0f}s)", flush=True)

    truth = np.array([f"{d}:{t}" for d, t in zip(articles["day"], articles["truth"])])
    preds = {name: np.array([f"{d}:{p}" for d, p in zip(articles["day"], out)])
             for (name, _, _), out in zip(CONFIGS, outputs)}

    table = pd.DataFrame({name: summary(truth, p) for name, p in preds.items()}).T
    table["mixed"] = [mixed_events(truth, p) for p in preds.values()]
    with pd.option_context("display.width", 220, "display.float_format", lambda x: f"{x:.3f}"):
        print(table[["bcubed_p", "bcubed_r", "bcubed_f1", "pair_p", "pair_r", "pair_f1",
                     "events_pred", "mixed", "largest_pred"]].to_string())

    base = "v2 continuous"
    pairwise_ci = bootstrap_pairwise(truth, preds)
    bcubed_ci = bootstrap_bcubed(truth, preds)
    print("\n== criteria (vs v2 continuous; 95% cluster-bootstrap interval of the difference)")
    for name, cap, merge in CONFIGS[1:]:
        pw = pairwise_ci["diff"].get((base, name))
        bc = bcubed_ci["diff"].get((base, name))
        # intervals are base - variant; flip to variant - base
        pw = (-pw[1], -pw[0])
        bc = (-bc[1], -bc[0])
        mixed_ok = table.loc[name, "mixed"] <= table.loc[base, "mixed"]
        if merge:
            ok = pw[0] > 0 and bc[1] >= 0 and table.loc[name, "bcubed_f1"] >= table.loc[base, "bcubed_f1"] and mixed_ok
        else:
            ok = pw[1] >= 0 and bc[1] >= 0 and mixed_ok
        print(f"  {name}: pairwise F1 diff {pw[0]:+.3f}..{pw[1]:+.3f}, B-cubed F1 diff {bc[0]:+.3f}..{bc[1]:+.3f}, "
              f"mixed {table.loc[name, 'mixed']} vs {table.loc[base, 'mixed']} -> {'PASS' if ok else 'FAIL'}")
    print(f"({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
