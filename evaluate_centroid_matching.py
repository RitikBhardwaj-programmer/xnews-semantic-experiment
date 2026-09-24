"""
Re-validates the event-match threshold and model against article-vs-event-
CENTROID features (V3 plan step 6). The original model was trained on
article-vs-article pairs, so its 0.70 threshold is not guaranteed to be right
for the similarities the centroid pipeline actually produces.

The labelled pairs are replayed as a stream, oldest first, mirroring the Java
pipeline: an event's centroid is a normalized running mean, temporal_score
uses the same day buckets against the event's last activity, and the top-30
nearest events are scored. Each article should either attach to its true
event or (if that event doesn't exist yet) create a new one. After each
decision the article is added to its TRUE event, so one mistake doesn't
corrupt every later step.

Usage:
    python evaluate_centroid_matching.py [--cache embeddings.npz]
    python evaluate_centroid_matching.py --write-model models/event_matcher.pkl
"""

import argparse
import glob
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TOP_K = 30
WINDOW_DAYS = 10
PRECISION_FLOOR = 0.95
FEATURES = ["similarity", "temporal_score"]
MODEL_PATH = Path(__file__).parent / "models" / "event_matcher.pkl"
THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.01), 2)


# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

def load_pairs():
    return pd.concat(
        [pd.read_csv(f) for f in sorted(glob.glob("data/*.csv"))],
        ignore_index=True
    )


def build_articles(pairs):

    articles = {}

    for side in ("a", "b"):
        for row in pairs.itertuples(index=False):

            article_id = getattr(row, f"article_{side}_id")

            articles.setdefault(article_id, {
                "id": article_id,
                "id_event": getattr(row, f"event_{side}_id"),
                "text": (
                    getattr(row, f"article_{side}_title")
                    + "\n"
                    + getattr(row, f"article_{side}_text")
                ),
                "date": np.datetime64(getattr(row, f"date_{side}"), "D")
            })

    return articles


def label_events(pairs):
    """Events as the model was trained to see them: SAME_EVENT links
    (including follow-ups) are merged with union-find."""

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for row in pairs.itertuples(index=False):
        find(row.article_a_id)
        find(row.article_b_id)

        if row.ground_truth == "SAME_EVENT":
            parent[find(row.article_a_id)] = find(row.article_b_id)

    return {article_id: find(article_id) for article_id in list(parent)}


def embed(articles, cache):

    ids = sorted(articles)

    if cache and Path(cache).exists():

        stored = np.load(cache, allow_pickle=True)

        if list(stored["ids"]) == ids:
            return ids, stored["embeddings"]

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    embeddings = model.encode(
        [articles[i]["text"] for i in ids],
        normalize_embeddings=True,
        batch_size=64
    )

    if cache:
        np.savez(cache, ids=np.array(ids, dtype=object), embeddings=embeddings)

    return ids, embeddings


# ------------------------------------------------------------
# PIPELINE MIRRORS (must match the Java side)
# ------------------------------------------------------------

def temporal_score(gap_days):

    gap = np.abs(gap_days)

    return np.select(
        [gap == 0, gap <= 1, gap <= 3, gap <= 7, gap <= 30],
        [1.0, 0.9, 0.8, 0.6, 0.4],
        default=0.1
    )


def normalize(vector):

    norm = np.linalg.norm(vector)

    return vector if norm == 0 else vector / norm


def update_centroid(centroid, count, embedding):
    """RunningMeanCentroidStrategy: normalize(centroid * n + embedding)."""

    return normalize(embedding) if centroid is None \
        else normalize(centroid * count + embedding)


# ------------------------------------------------------------
# SIMULATION
# ------------------------------------------------------------

def simulate(ids, embeddings, articles, truth, window):

    days = np.array(
        [articles[i]["date"].astype("int64") for i in ids]
    )

    order = sorted(range(len(ids)), key=lambda k: (days[k], ids[k]))

    events, centroids, counts, last = [], [], [], []
    position = {}

    article_rows, candidate_rows = [], []

    for k in order:

        event = truth[ids[k]]
        true_position = position.get(event)

        row = {"k": k, "event": event,
               "attach_case": true_position is not None,
               "expected_attach": False}

        if events:

            matrix = np.vstack(centroids)
            gap = days[k] - np.array(last)

            pool = np.arange(len(events)) if window is None \
                else np.flatnonzero(gap <= window)

            if len(pool):

                sims = matrix[pool] @ embeddings[k]
                best = np.argsort(-sims)[:TOP_K]
                chosen = pool[best]
                temporal = temporal_score(gap[chosen])

                for p, sim, t in zip(chosen, sims[best], temporal):
                    candidate_rows.append(
                        (k, events[p], float(sim), float(t), events[p] == event)
                    )

                row["expected_attach"] = (
                    true_position is not None and true_position in set(pool)
                )

        article_rows.append(row)

        # oracle update: keeps the store clean of earlier mistakes
        if true_position is None:
            position[event] = len(events)
            events.append(event)
            centroids.append(update_centroid(None, 0, embeddings[k]))
            counts.append(1)
            last.append(days[k])
        else:
            centroids[true_position] = update_centroid(
                centroids[true_position], counts[true_position], embeddings[k]
            )
            counts[true_position] += 1
            last[true_position] = max(last[true_position], days[k])

    articles_df = pd.DataFrame(article_rows).set_index("k")

    candidates_df = pd.DataFrame(
        candidate_rows,
        columns=["k", "event", "similarity", "temporal_score", "is_true"]
    )

    candidates_df = candidates_df.merge(
        articles_df[["event"]].rename(columns={"event": "article_event"}),
        left_on="k", right_index=True
    )

    return articles_df, candidates_df


# ------------------------------------------------------------
# SCORING
# ------------------------------------------------------------

def sweep(articles_df, candidates_df, score):
    """Attach/create outcome metrics for every threshold, for one score."""

    best = candidates_df.loc[
        candidates_df.groupby("k")[score].idxmax()
    ].set_index("k")

    frame = articles_df.join(
        best[[score, "is_true"]].rename(
            columns={score: "best_score", "is_true": "best_true"}
        )
    )

    frame["best_score"] = frame["best_score"].fillna(-np.inf)
    frame["best_true"] = frame["best_true"].fillna(False).astype(bool)

    expected = frame["expected_attach"].to_numpy()
    good_pick = frame["best_true"].to_numpy() & expected
    scores = frame["best_score"].to_numpy()

    rows = []

    for t in THRESHOLDS:

        attached = scores >= t
        true_positive = (attached & good_pick).sum()
        wrong = (attached & ~good_pick).sum()
        missed = (expected & ~attached).sum()

        precision = true_positive / max(true_positive + wrong, 1)
        recall = true_positive / max(expected.sum(), 1)
        f1 = 0 if precision + recall == 0 else \
            2 * precision * recall / (precision + recall)

        rows.append({
            "threshold": t, "precision": precision, "recall": recall,
            "f1": f1, "wrong_merges": int(wrong), "duplicates": int(missed)
        })

    return pd.DataFrame(rows), float(good_pick.sum() / max(expected.sum(), 1))


def pick(table):
    """Best F1, preferring thresholds that keep precision above the floor.
    Returns (row, whether the floor was reachable)."""

    ok = table[table["precision"] >= PRECISION_FLOOR]
    pool = ok if len(ok) else table

    row = pool.sort_values(["f1", "threshold"], ascending=[False, False]).iloc[0]

    return row, len(ok) > 0


def plateau_threshold(table, tolerance=0.01):
    """Middle of the thresholds whose F1 is within `tolerance` of the best,
    so the choice isn't a knife-edge."""

    near = table[table["f1"] >= table["f1"].max() - tolerance]

    return float(np.median(near["threshold"]))


def show(title, row, note=""):

    print(
        f"  {title:<26} threshold={row.threshold:.2f}  "
        f"precision={row.precision:.3f}  recall={row.recall:.3f}  "
        f"F1={row.f1:.3f}  wrong merges={int(row.wrong_merges)}  "
        f"duplicates={int(row.duplicates)}{note}"
    )


def new_pipeline(cols):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000))
    ])


def out_of_fold(c, cols):
    """Scores from a model that never saw the event it is scoring."""

    scores = np.zeros(len(c))

    for train, test in GroupKFold(n_splits=5).split(c, groups=c["article_event"]):
        model = new_pipeline(cols)
        model.fit(c.iloc[train][cols], c.iloc[train]["is_true"])
        scores[test] = model.predict_proba(c.iloc[test][cols])[:, 1]

    return scores


def coefficients(model):
    return np.round(model.steps[-1][1].coef_[0], 2)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=None)
    parser.add_argument("--write-model", default=None,
                        help="fit the recommended model and save it here")
    args = parser.parse_args()

    pairs = load_pairs()
    articles = build_articles(pairs)
    label_truth = label_events(pairs)
    id_truth = {a: articles[a]["id_event"] for a in articles}
    old_model = joblib.load(MODEL_PATH)

    ids, embeddings = embed(articles, args.cache)

    print(f"{len(ids)} articles | {len(set(label_truth.values()))} label-events "
          f"| {len(set(id_truth.values()))} strict events (follow-ups separate)")

    # -- 1. how much did the similarity distribution move? ----
    index = {a: n for n, a in enumerate(ids)}
    sim_pairs = np.array([
        embeddings[index[r.article_a_id]] @ embeddings[index[r.article_b_id]]
        for r in pairs.itertuples(index=False)
    ])
    same = (pairs["ground_truth"] == "SAME_EVENT").to_numpy()

    _, cands = simulate(ids, embeddings, articles, label_truth, None)
    true_c = cands[cands["is_true"]]["similarity"]
    other_c = cands[~cands["is_true"]]["similarity"]

    def pct(series, qs):
        return "  ".join(f"p{q}={np.percentile(series, q):.2f}" for q in qs)

    print("\n1. SIMILARITY DISTRIBUTION SHIFT (cosine)")
    print("  article vs article, SAME event      ", pct(sim_pairs[same], (10, 50, 90)))
    print("  article vs article, DIFFERENT event ", pct(sim_pairs[~same], (50, 90, 99)))
    print("  article vs its event centroid       ", pct(true_c, (10, 50, 90)))
    print("  article vs other event centroids    ", pct(other_c, (50, 90, 99)))

    # -- 2. threshold sweep under each set of assumptions -----
    print(f"\n2. THRESHOLD FOR THE CURRENT MODEL (best F1; precision floor "
          f"{PRECISION_FLOOR} is used only where reachable)")

    settings = []

    for truth_name, truth in (("label events", label_truth),
                              ("strict events", id_truth)):
        for window in (None, WINDOW_DAYS):

            a, c = simulate(ids, embeddings, articles, truth, window)
            c["old_model"] = old_model.predict_proba(c[FEATURES])[:, 1]

            table, top1 = sweep(a, c, "old_model")
            label = f"{truth_name}, window={'none' if window is None else str(window) + 'd'}"
            settings.append((label, a, c))

            print(f"\n  [{label}]  attach cases={int(a['expected_attach'].sum())}  "
                  f"articles={len(a)}  true event ranked #1: {top1:.3f}")
            show("current 0.70", table[table.threshold == 0.70].iloc[0])

            row, met = pick(table)
            show("best", row, "" if met else "   (precision floor unreachable)")

    # -- 3. does the classifier earn its place? ---------------
    print("\n3. CLASSIFIER vs PLAIN SIMILARITY (retrained = 5-fold, split by event)")

    production_like = None

    for label, a, c in settings:

        print(f"\n  [{label}]")

        c["similarity_only"] = c["similarity"]
        c["retrained_sim_temporal"] = out_of_fold(c, FEATURES)
        c["retrained_sim_only"] = out_of_fold(c, ["similarity"])

        for name in ("old_model", "similarity_only",
                     "retrained_sim_temporal", "retrained_sim_only"):

            table, top1 = sweep(a, c, name)
            row, _ = pick(table)
            ap = average_precision_score(c["is_true"], c[name])
            print(f"    {name:<24} AP={ap:.3f}  top-1={top1:.3f}  "
                  f"best F1={row.f1:.3f} @ {row.threshold:.2f} "
                  f"(precision {row.precision:.3f}, recall {row.recall:.3f})")

        full = new_pipeline(FEATURES).fit(c[FEATURES], c["is_true"])
        print("    coefficients [similarity, temporal]: old model",
              coefficients(old_model), "| retrained", coefficients(full))

        if label == f"label events, window={WINDOW_DAYS}d":
            production_like = (a, c)

    # -- 4. how much does time alone buy? ---------------------
    print("\n4. LOWEST SIMILARITY AT WHICH A MODEL REACHES A PROBABILITY")
    print("   (candidates inside the lifecycle window all have temporal_score >= 0.6)")

    a, c = production_like

    recommended = new_pipeline(FEATURES).fit(c[FEATURES], c["is_true"])

    grid = np.round(np.arange(0.0, 1.0001, 0.01), 2)

    for name, model in (("current model", old_model), ("retrained model", recommended)):

        print(f"   {name}")
        print(f"   {'temporal_score':<16}{'>=0.70':>8}{'>=0.83':>8}{'>=0.90':>8}{'>=0.95':>8}")

        for temporal in (1.0, 0.9, 0.8, 0.6):
            p = model.predict_proba(
                pd.DataFrame({"similarity": grid, "temporal_score": temporal})
            )[:, 1]
            cells = [f"{grid[p >= t][0]:.2f}" if (p >= t).any() else "never"
                     for t in (0.70, 0.83, 0.90, 0.95)]
            print(f"   {temporal:<16}" + "".join(f"{cell:>8}" for cell in cells))

    # -- 5. the recommendation --------------------------------
    scores = out_of_fold(c, FEATURES)
    c = c.assign(candidate=scores)
    table, _ = sweep(a, c, "candidate")
    chosen = table.iloc[(table["threshold"] - plateau_threshold(table)).abs().argmin()]
    threshold = float(chosen.threshold)

    print(f"\n5. RECOMMENDATION (label events, {WINDOW_DAYS}-day window, retrained "
          f"on similarity + temporal_score)")
    show("cross-validated", chosen)
    print("  F1 across the plateau:  " + "  ".join(
        f"{t:.2f}:{table[table.threshold == t].iloc[0].f1:.3f}"
        for t in (0.80, 0.85, 0.90, 0.93, 0.95, 0.97)))

    if args.write_model:

        out = Path(args.write_model)
        out.parent.mkdir(exist_ok=True)
        joblib.dump(recommended, out)

        meta = {
            "trained_at": datetime.now().isoformat(),
            "source_file": "evaluate_centroid_matching.py",
            "trained_on": (f"article-vs-event-centroid rows, label events, "
                           f"{WINDOW_DAYS}-day window, top {TOP_K} candidates"),
            "features": FEATURES,
            "training_rows": int(len(c)),
            "attach_cases": int(a["expected_attach"].sum()),
            "recommended_threshold": round(threshold, 2),
            "cross_validated_precision": round(float(chosen.precision), 3),
            "cross_validated_recall": round(float(chosen.recall), 3),
            "cross_validated_f1": round(float(chosen.f1), 3)
        }

        out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

        print(f"\n  wrote {out} and {out.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
