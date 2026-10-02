"""
Stage 1 of V4 approach 1: does event evidence beat the one-number matcher?

Leave-one-day-out over the labelled days. For each fold:
1. Training rows come from an ORACLE replay of the two training days
   (every article joins its true event), one row per (article, candidate event)
   with the features from event_features.py and is_true.
2. A model is fitted on those rows; its threshold is chosen on the training
   days only (best attach-decision F1 on the training rows).
3. The held-out day is replayed FREE-RUNNING with the new decision and scored.

Variants are added one piece at a time (ablation) and compared with
replay@0.94, the production rule, on the same folds. Paired cluster-bootstrap
intervals resample true events.

Usage:
    python evaluate_stage1.py                 (all variants)
    python evaluate_stage1.py --quick         (fewer variants, for a smoke run)
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cluster_metrics import bootstrap_bcubed, bootstrap_pairwise, summary
from event_features import FEATURE_GROUPS, ArticleIndex, strip_templates
from replay import THRESHOLD, load_day, production_replay, replay

BASE = Path(__file__).parent / "data" / "validation" / "eval_days"
DAYS = ["2026-09-29", "2026-09-30", "2026-10-01"]
MEMBER_GATE = 0.45
THRESHOLDS = np.round(np.concatenate([np.arange(0.30, 0.90, 0.05), np.arange(0.90, 0.995, 0.01)]), 2)


# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

def load(day, stripped_vectors=None):
    articles = load_day(BASE / f"articles_{day}.csv")
    descriptions = pd.read_csv(BASE / "local" / "descriptions.csv").set_index("id")["description"]
    articles["description"] = articles["id"].map(descriptions)
    labels = pd.read_csv(BASE / f"labels_{day}.csv").set_index("article_id")["event_label"]
    articles["truth"] = articles["id"].map(labels)
    articles["stripped"] = articles["title"].map(strip_templates)
    if stripped_vectors is not None:
        articles["vector"] = list(np.stack([stripped_vectors[i] for i in articles["id"]]))
    return articles


def stripped_embeddings(frames):
    """Re-embed stripped title + description with the locally cached MiniLM (cached to local/)."""
    cache = BASE / "local" / "stripped_embeddings.npz"
    if cache.exists():
        data = np.load(cache)
        return dict(zip(data["ids"], data["vectors"]))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    allf = pd.concat(frames)
    texts = (allf["stripped"] + "\n" + allf["description"].fillna("")).tolist()
    vectors = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    np.savez(cache, ids=allf["id"].to_numpy(), vectors=vectors)
    return dict(zip(allf["id"], vectors))


def build_indexes(frames):
    """TF-IDF fitted on all days' text (no labels used), then one ArticleIndex per day."""
    titles = TfidfVectorizer(sublinear_tf=True, min_df=1, ngram_range=(1, 2), stop_words="english")
    texts = TfidfVectorizer(sublinear_tf=True, min_df=1, stop_words="english")
    allf = pd.concat(frames)
    titles.fit(allf["stripped"])
    texts.fit(allf["stripped"] + " " + allf["description"].fillna(""))
    return [
        ArticleIndex(f.assign(title=f["stripped"]), titles.transform(f["stripped"]),
                     texts.transform(f["stripped"] + " " + f["description"].fillna("")))
        for f in frames
    ]


# ------------------------------------------------------------
# TRAINING ROWS (oracle replay)
# ------------------------------------------------------------

def oracle_rows(articles, index):
    """Every article joins its TRUE event; record features for each candidate."""
    rows, event_of_truth = [], {}

    def decide(article, candidates):
        row, truth = article.name, article["truth"]
        true_event = event_of_truth.get(truth)
        for c in candidates:
            features = index.features(row, c["members"], c["similarity"], c["temporal_score"])
            features.update(article=row, event=c["index"], is_true=c["index"] == true_event,
                            conflict=index.conflicts(row, c["members"]))
            rows.append(features)
        if true_event is None:
            event_of_truth[truth] = decide.created
            decide.created += 1
        return true_event

    decide.created = 0
    replay(articles, decide)
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# MODELS AND DECISIONS
# ------------------------------------------------------------

def fit(rows, columns, kind):
    if kind == "hgb":
        model = CalibratedClassifierCV(
            HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05),
            method="sigmoid", cv=3,
        )
    else:
        model = Pipeline([("scale", StandardScaler()),
                          ("lr", LogisticRegression(class_weight="balanced", max_iter=2000))])
    return model.fit(rows[columns], rows["is_true"])


def choose_threshold(model, rows, columns, veto):
    """Best attach-decision F1 on the TRAINING rows: per article, attach to the best
    candidate if its probability >= t; correct when that candidate is the true event."""
    rows = rows.assign(p=model.predict_proba(rows[columns])[:, 1])
    if veto:
        rows = rows[~rows["conflict"]]
    best = rows.loc[rows.groupby("article")["p"].idxmax()]
    has_true = rows.groupby("article")["is_true"].any()
    scores = []
    for t in THRESHOLDS:
        attach = best["p"] >= t
        correct = (attach & best["is_true"]).sum()
        precision = correct / attach.sum() if attach.sum() else 0
        recall = correct / has_true.sum() if has_true.sum() else 0
        scores.append((2 * precision * recall / (precision + recall) if precision + recall else 0, t))
    return max(scores)[1]


def model_decide(model, columns, threshold, index, veto, gate):
    def decide(article, candidates):
        row = article.name
        options = []
        for c in candidates:
            if veto and index.conflicts(row, c["members"]):
                continue
            features = index.features(row, c["members"], c["similarity"], c["temporal_score"])
            if gate and features["member_min"] < MEMBER_GATE:
                continue
            options.append((c["index"], features))
        if not options:
            return None
        frame = pd.DataFrame([f for _, f in options])[columns]
        p = model.predict_proba(frame)[:, 1]
        best = int(np.argmax(p))
        return options[best][0] if p[best] >= threshold else None
    return decide


def gated_production(index):
    """Production rule plus a hard member gate (min cosine to members >= 0.45)."""
    import joblib
    from evaluate_centroid_matching import FEATURES, MODEL_PATH
    model = joblib.load(MODEL_PATH)

    def decide(article, candidates):
        row = article.name
        options = [c for c in candidates
                   if index.features(row, c["members"], c["similarity"], c["temporal_score"])["member_min"] >= MEMBER_GATE]
        if not options:
            return None
        frame = pd.DataFrame([(c["similarity"], c["temporal_score"]) for c in options], columns=FEATURES)
        p = model.predict_proba(frame)[:, 1]
        best = int(np.argmax(p))
        return options[best]["index"] if p[best] >= THRESHOLD else None
    return decide


# ------------------------------------------------------------
# EXPERIMENT
# ------------------------------------------------------------

VARIANTS = [
    # name, feature groups, veto, gate, model, stripped embeddings
    ("A members", ["base", "members"], False, False, "lr", False),
    ("B +lexical", ["base", "members", "lexical"], False, False, "lr", False),
    ("C +entities", ["base", "members", "lexical", "entities"], False, False, "lr", False),
    ("D +veto", ["base", "members", "lexical", "entities"], True, False, "lr", False),
    ("E +time (full)", ["base", "members", "lexical", "entities", "time"], True, False, "lr", False),
    ("F full, stripped re-embed", ["base", "members", "lexical", "entities", "time"], True, False, "lr", True),
    ("G full, HGB", ["base", "members", "lexical", "entities", "time"], True, False, "hgb", False),
    ("H full + hard gate", ["base", "members", "lexical", "entities", "time"], True, True, "lr", False),
]


def columns_for(groups):
    return [c for g in groups for c in FEATURE_GROUPS[g]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    variants = VARIANTS[:2] if args.quick else VARIANTS
    started = time.time()

    frames = {False: [load(d) for d in DAYS]}
    if any(v[5] for v in variants):
        vectors = stripped_embeddings(frames[False])
        frames[True] = [load(d, vectors) for d in DAYS]
    indexes = {k: build_indexes(f) for k, f in frames.items()}
    print(f"loaded {len(DAYS)} days in {time.time() - started:.0f}s")

    # training rows per (embedding, day), computed once
    rows = {k: [oracle_rows(f, i) for f, i in zip(frames[k], indexes[k])] for k in frames}
    print(f"oracle rows: {[len(r) for r in rows[False]]} ({time.time() - started:.0f}s)")

    truth = np.concatenate([f["truth"].to_numpy() for f in frames[False]])
    preds = {"replay@0.94": [], "gate only (prod + min>=0.45)": []}
    for f, i in zip(frames[False], indexes[False]):
        preds["replay@0.94"].append(production_replay(f))
        preds["gate only (prod + min>=0.45)"].append(replay(f, gated_production(i)))
    print(f"baselines done ({time.time() - started:.0f}s)")

    report = {"thresholds": {}, "coefficients": {}}
    for name, groups, veto, gate, kind, stripped in variants:
        columns = columns_for(groups)
        preds[name] = []
        for test, day in enumerate(DAYS):
            train = pd.concat([r for d, r in enumerate(rows[stripped]) if d != test])
            model = fit(train, columns, kind)
            threshold = choose_threshold(model, train, columns, veto)
            report["thresholds"].setdefault(name, []).append(float(threshold))
            if kind == "lr" and test == 0:
                report["coefficients"][name] = dict(zip(columns, np.round(model[-1].coef_[0], 2).tolist()))
            decide = model_decide(model, columns, threshold, indexes[stripped][test], veto, gate)
            preds[name].append(replay(frames[stripped][test], decide))
        print(f"{name}: thresholds {report['thresholds'][name]} ({time.time() - started:.0f}s)")

    # scoring: labels are per day, so prefix predictions by day before pooling
    pooled = {n: np.concatenate([[f"{d}:{x}" for x in p] for d, p in zip(DAYS, ps)]) for n, ps in preds.items()}
    truth_p = np.concatenate([[f"{d}:{x}" for x in f["truth"]] for d, f in zip(DAYS, frames[False])])
    columns = ["bcubed_p", "bcubed_r", "bcubed_f1", "pair_p", "pair_r", "pair_f1", "ceafe_f1", "events_pred", "largest_pred"]
    table = pd.DataFrame({n: summary(truth_p, p) for n, p in pooled.items()}).T[columns]
    with pd.option_context("display.width", 220, "display.float_format", lambda x: f"{x:.3f}"):
        print("\n== pooled over 3 held-out days (each day scored by a model trained on the other two)")
        print(table.to_string())
        for d, day in enumerate(DAYS):
            per = pd.DataFrame({n: summary(frames[False][d]["truth"], ps[d]) for n, ps in preds.items()}).T
            print(f"\n-- {day}")
            print(per[["bcubed_f1", "pair_p", "pair_r", "pair_f1", "events_pred", "largest_pred"]].to_string())

    diffs_b = bootstrap_bcubed(truth_p, pooled)["diff"]
    diffs_p = bootstrap_pairwise(truth_p, pooled)["diff"]
    print("\n== paired 95% intervals vs replay@0.94 (resampling true events)")
    for name in pooled:
        if name == "replay@0.94":
            continue
        b = diffs_b[("replay@0.94", name)]
        p = diffs_p[("replay@0.94", name)]
        # intervals are (baseline - variant); flip sign to read as (variant - baseline)
        print(f"  {name:<30} pairwise F1 [{-p[1]:+.3f}, {-p[0]:+.3f}]   B-cubed F1 [{-b[1]:+.3f}, {-b[0]:+.3f}]")

    print("\n== LR coefficients (fold 1, standardised features)")
    for name, coefs in report["coefficients"].items():
        print(f"  {name}: {coefs}")

    out = BASE / "local" / "stage1_predictions.json"
    json.dump({n: [list(map(int, p)) for p in ps] for n, ps in preds.items()}, open(out, "w"))
    print(f"\ndone in {time.time() - started:.0f}s; predictions saved to {out}")


if __name__ == "__main__":
    main()
