"""
Stage 1b step 3: evaluate the v2 matcher's shadow run in production against
the go-live criteria fixed in the stage 1b plan.

1. Export (read-only) the shadow window from production:
       python evaluate_shadow.py export-sql --from "2026-10-02 07:43" --to "2026-10-05 07:45"
   writes data/validation/shadow/local/export.sql; run it with psql
   (the xnews-prod-db skill) from that folder. Everything lands in local/
   (git-ignored): it contains publishers' descriptions.

2. Evaluate:
       python evaluate_shadow.py evaluate
   - v2 health: error rate and latency
   - agreement between v1 (applied) and v2 (shadow)
   - disagreements.csv for labelling (EVENT_DEFINITION.md): fill `label`
     with v1, v2, neither or unsure, save as disagreements_labelled.csv
   - would-be event sizes: a free-running v2 replay of the whole window
     (continuous, so events can span days), vocabulary refitted nightly on
     the previous 14 days as in production

Go-live criteria (all must pass):
   - v2 right in significantly more labelled disagreements than v1
     (95% Wilson interval of v2's share, among cases where exactly one is
     right, entirely above 50%)
   - v2 error rate < 1%
   - p95 v2 latency < 1.5 s
   - no event in the v2 replay larger than 2x the largest true event in the
     labelled days
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from event_features import new_vectorizers, strip_templates, tfidf_text
from evaluate_stage1 import RefitIndex, model_decide
from matcher_v2 import MEMBER_TEXT_CAP, load_matcher
from replay import replay

BASE = Path(__file__).parent / "data" / "validation"
LOCAL = BASE / "shadow" / "local"
LABELLED_DAYS = ["2026-09-29", "2026-09-30", "2026-10-01"]
VOCABULARY_DAYS = 14

MAX_ERROR_RATE = 0.01
MAX_P95_MS = 1500
MAX_SIZE_FACTOR = 2


# ------------------------------------------------------------
# EXPORT
# ------------------------------------------------------------

def export_sql(start, end):
    window = f"created_at >= '{start}' and created_at < '{end}'"
    decided = f"select article_id from event_match_decisions where {window}"
    return f"""-- Read-only export of the shadow window {start} .. {end} (UTC)
\\copy (select article_id, matcher, mode, model_version, chosen_event_id, applied, probability, threshold, candidates::text as candidates, latency_ms, error, created_at from event_match_decisions where {window} order by id) to 'decisions.csv' csv header
\\copy (select id, source, title, description, published_at, created_at, news_event_id, embedding::text as embedding from articles where id in ({decided}) order by created_at, id) to 'articles.csv' csv header
\\copy (select id, news_event_id, title, created_at from articles where news_event_id in (select chosen_event_id from event_match_decisions where {window} and chosen_event_id is not null)) to 'members.csv' csv header
\\copy (select id, title, description, created_at from articles where created_at >= timestamp '{start}' - interval '{VOCABULARY_DAYS} days' and created_at < '{end}') to 'vocab_texts.csv' csv header
"""


# ------------------------------------------------------------
# HEALTH AND AGREEMENT
# ------------------------------------------------------------

def health(decisions):
    v2 = decisions[decisions["matcher"] == "v2"]
    errors = v2["error"].notna().sum()
    latency = v2.loc[v2["error"].isna(), "latency_ms"].dropna()
    return {
        "v2_rows": len(v2),
        "v2_errors": int(errors),
        "error_rate": errors / len(v2) if len(v2) else float("nan"),
        "p50_ms": float(latency.quantile(0.5)) if len(latency) else float("nan"),
        "p95_ms": float(latency.quantile(0.95)) if len(latency) else float("nan"),
    }


def paired(decisions):
    """One row per article with both decisions; v2 rows that errored are left out."""
    v1 = decisions[decisions["matcher"] == "v1"].set_index("article_id")
    v2 = decisions[(decisions["matcher"] == "v2") & decisions["error"].isna()].set_index("article_id")
    both = v1[["chosen_event_id", "probability"]].join(
        v2[["chosen_event_id", "probability", "candidates"]], lsuffix="_v1", rsuffix="_v2", how="inner")
    both["same"] = [a == b or (pd.isna(a) and pd.isna(b))
                    for a, b in zip(both["chosen_event_id_v1"], both["chosen_event_id_v2"])]
    return both


# ------------------------------------------------------------
# DISAGREEMENTS (for labelling)
# ------------------------------------------------------------

def member_titles(members, event_id, before, limit=6):
    if pd.isna(event_id):
        return "(new event)"
    rows = members[(members["news_event_id"] == event_id) & (members["created_at"] < before)]
    rows = rows.sort_values("created_at", ascending=False)
    titles = " // ".join(rows["title"].head(limit))
    return f"[{len(rows)}] {titles}"


def disagreements(pairs, articles, members):
    rows = []
    for article_id, pair in pairs[~pairs["same"]].iterrows():
        article = articles.loc[article_id]
        rows.append({
            "article_id": article_id,
            "outlet": article["source"],
            "published_at": article["published_at"],
            "title": article["title"],
            "v1_event_id": pair["chosen_event_id_v1"],
            "v1_probability": pair["probability_v1"],
            "v1_event": member_titles(members, pair["chosen_event_id_v1"], article["created_at"]),
            "v2_event_id": pair["chosen_event_id_v2"],
            "v2_probability": pair["probability_v2"],
            "v2_event": member_titles(members, pair["chosen_event_id_v2"], article["created_at"]),
            "label": "",
            "note": "",
        })
    return pd.DataFrame(rows)


def wilson(successes, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def label_result(labelled):
    counts = labelled["label"].str.strip().str.lower().value_counts()
    v1, v2 = int(counts.get("v1", 0)), int(counts.get("v2", 0))
    low, high = wilson(v2, v1 + v2)
    return {"v1_right": v1, "v2_right": v2, "neither": int(counts.get("neither", 0)),
            "unsure": int(counts.get("unsure", 0)), "v2_share": v2 / (v1 + v2) if v1 + v2 else float("nan"),
            "interval": (low, high)}


# ------------------------------------------------------------
# V2 REPLAY (would-be event sizes)
# ------------------------------------------------------------

def nightly_index(articles, vocab_texts):
    """Each article is scored with vocabulary fitted on the 14 days before its UTC day."""
    days = articles["created_at"].dt.floor("D")
    codes, starts = pd.factorize(days)
    stripped = articles["title"].map(strip_templates)
    texts = [tfidf_text(s, d if isinstance(d, str) else None) for s, d in zip(stripped, articles["description"])]
    matrices = {}
    for code, start in enumerate(starts):
        seen = vocab_texts[(vocab_texts["created_at"] < start)
                           & (vocab_texts["created_at"] >= start - pd.Timedelta(days=VOCABULARY_DAYS))]
        seen_stripped = seen["title"].map(strip_templates)
        titles, text_vectorizer = new_vectorizers()
        titles.fit(seen_stripped)
        text_vectorizer.fit([tfidf_text(s, d if isinstance(d, str) else None)
                             for s, d in zip(seen_stripped, seen["description"])])
        matrices[code] = (titles.transform(stripped), text_vectorizer.transform(texts))
    return RefitIndex(articles.assign(title=stripped), matrices, codes, MEMBER_TEXT_CAP)


def v2_replay(articles, vocab_texts):
    matcher = load_matcher()
    index = nightly_index(articles, vocab_texts)
    return replay(articles, model_decide(matcher.model, matcher.columns, matcher.threshold, index, False, False))


def largest_true_event():
    return max(pd.read_csv(BASE / "eval_days" / f"labels_{d}.csv")["event_label"].value_counts().max()
               for d in LABELLED_DAYS)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def load_export(folder):
    decisions = pd.read_csv(folder / "decisions.csv", parse_dates=["created_at"])
    articles = pd.read_csv(folder / "articles.csv", parse_dates=["published_at", "created_at"])
    vectors = np.stack(articles["embedding"].map(
        lambda s: np.array([float(x) for x in s.strip("[]").split(",")], dtype=np.float64)).values)
    articles["vector"] = list(vectors / np.linalg.norm(vectors, axis=1, keepdims=True))
    articles = articles.drop(columns=["embedding"]).sort_values(["created_at", "id"]).reset_index(drop=True)
    members = pd.read_csv(folder / "members.csv", parse_dates=["created_at"])
    vocab_texts = pd.read_csv(folder / "vocab_texts.csv", parse_dates=["created_at"])
    return decisions, articles, members, vocab_texts


def evaluate(folder, skip_replay=False):
    decisions, articles, members, vocab_texts = load_export(folder)
    verdicts = {}

    h = health(decisions)
    print(f"window: {decisions['created_at'].min()} .. {decisions['created_at'].max()}, "
          f"{decisions['article_id'].nunique()} articles")
    print(f"v2 health: {h['v2_errors']} errors in {h['v2_rows']} calls ({h['error_rate']:.2%}), "
          f"latency p50 {h['p50_ms']:.0f} ms, p95 {h['p95_ms']:.0f} ms")
    errors = decisions.loc[decisions["error"].notna(), ["created_at", "error"]]
    for kind, group in errors.groupby(errors["error"].str.slice(0, 60)):
        print(f"  {len(group)} x {kind}... (first {group['created_at'].min()}, last {group['created_at'].max()})")
    verdicts["v2 error rate < 1%"] = h["error_rate"] < MAX_ERROR_RATE
    verdicts["p95 v2 latency < 1.5 s"] = h["p95_ms"] < MAX_P95_MS

    pairs = paired(decisions)
    attach_v1 = pairs["chosen_event_id_v1"].notna().sum()
    attach_v2 = pairs["chosen_event_id_v2"].notna().sum()
    print(f"agreement: {pairs['same'].sum()}/{len(pairs)} ({pairs['same'].mean():.1%}); "
          f"attached v1 {attach_v1}, v2 {attach_v2}")

    table = disagreements(pairs, articles.set_index("id"), members)
    table.to_csv(folder / "disagreements.csv", index=False)
    print(f"disagreements: {len(table)} written to {folder / 'disagreements.csv'}")

    labelled_path = folder / "disagreements_labelled.csv"
    if labelled_path.exists():
        r = label_result(pd.read_csv(labelled_path, keep_default_na=False))
        low, high = r["interval"]
        print(f"labelled: v2 right {r['v2_right']}, v1 right {r['v1_right']}, neither {r['neither']}, "
              f"unsure {r['unsure']}; v2 share {r['v2_share']:.1%} (95% {low:.1%} .. {high:.1%})")
        verdicts["v2 right in significantly more disagreements"] = low > 0.5
    else:
        print("labelled: not yet (fill `label` in disagreements.csv, save as disagreements_labelled.csv)")
        verdicts["v2 right in significantly more disagreements"] = None

    if not skip_replay:
        limit = MAX_SIZE_FACTOR * largest_true_event()
        predicted = v2_replay(articles, vocab_texts)
        sizes = pd.Series(predicted).value_counts()
        production_sizes = articles["news_event_id"].value_counts()
        print(f"v2 replay: {sizes.size} events, largest {sizes.head(5).tolist()} "
              f"(limit {limit}); production events in window, largest {production_sizes.head(5).tolist()}")
        verdicts[f"no v2 replay event over {limit}"] = int(sizes.max()) <= limit

    print("\n== go-live criteria")
    for name, ok in verdicts.items():
        print(f"  {'PASS' if ok else ('PENDING' if ok is None else 'FAIL')}: {name}")
    decided = [ok for ok in verdicts.values() if ok is not None]
    print("GO-LIVE: " + ("all criteria pass" if all(decided) and None not in verdicts.values()
                         else "not yet" if all(decided) else "criteria failed"))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export-sql")
    export.add_argument("--from", dest="start", required=True, help="UTC, e.g. '2026-10-02 07:43'")
    export.add_argument("--to", dest="end", required=True)
    run = sub.add_parser("evaluate")
    run.add_argument("--dir", type=Path, default=LOCAL)
    run.add_argument("--skip-replay", action="store_true")
    args = parser.parse_args()

    if args.command == "export-sql":
        LOCAL.mkdir(parents=True, exist_ok=True)
        (LOCAL / "export.sql").write_text(export_sql(args.start, args.end), encoding="utf-8")
        print(f"wrote {LOCAL / 'export.sql'}")
    else:
        evaluate(args.dir, args.skip_replay)


if __name__ == "__main__":
    main()
