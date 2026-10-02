"""
Baseline for V4 approach 1 (stage 0): how good is event grouping today,
measured against hand-labelled days (data/validation/eval_days/labels_<day>.csv,
made with label_events.py and EVENT_DEFINITION.md).

Systems compared on the same articles:
- production  : the event each article actually got in production
- replay@T    : free-running replay of the production matcher at threshold T
                (replay.py), for T in 0.90..0.99

Every system sees ALL articles of a day; scores use only the labelled ones,
so a partly labelled day still works. Days are scored separately and pooled
(labels never span days). Production event ids are prefixed by day when
pooling, so all systems are compared on the same per-day basis.

Usage:
    python evaluate_baseline.py --days 2026-09-29 [2026-09-30 2026-10-01]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cluster_metrics import bootstrap_bcubed, summary
from replay import THRESHOLD, load_day, production_replay

BASE = Path(__file__).parent / "data" / "validation" / "eval_days"
SWEEP = np.round(np.arange(0.90, 0.991, 0.01), 2)
COLUMNS = ["bcubed_p", "bcubed_r", "bcubed_f1", "pair_p", "pair_r", "pair_f1",
           "ceafe_f1", "ari", "events_pred", "events_true", "largest_pred"]


def labelling_minutes(labels, idle_minutes=5):
    """Active labelling time: gaps between saves, ignoring breaks."""
    times = pd.to_datetime(labels["labelled_at"]).sort_values()
    gaps = times.diff().dt.total_seconds().dropna() / 60
    return float(gaps[gaps <= idle_minutes].sum())


def load(day):

    labels_path = BASE / f"labels_{day}.csv"
    if not labels_path.exists():
        return None

    articles = load_day(BASE / f"articles_{day}.csv")
    labels = pd.read_csv(labels_path, keep_default_na=False)

    systems = {"production": articles["news_event_id"].astype(str).to_numpy()}
    for threshold in SWEEP:
        systems[f"replay@{threshold:.2f}"] = production_replay(articles, threshold).astype(str)

    labelled = articles["id"].isin(labels["article_id"]).to_numpy()
    truth = articles.loc[labelled, "id"].map(labels.set_index("article_id")["event_label"]).to_numpy()

    return {
        "day": day,
        "articles": len(articles),
        "labelled": int(labelled.sum()),
        "notes": int((labels["note"] != "").sum()),
        "minutes": labelling_minutes(labels),
        "truth": truth,
        "systems": {name: np.array([f"{day}:{x}" for x in pred[labelled]]) for name, pred in systems.items()},
    }


def table(truth, systems):
    rows = {name: summary(truth, pred) for name, pred in systems.items()}
    return pd.DataFrame(rows).T[COLUMNS]


def show(title, frame):
    print(f"\n{title}")
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", lambda x: f"{x:.3f}"):
        print(frame.to_string())


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--days", nargs="+", required=True)
    args = parser.parse_args()

    days = [d for d in (load(day) for day in args.days) if d is not None]
    missing = set(args.days) - {d["day"] for d in days}
    if missing:
        print(f"no labels yet for: {', '.join(sorted(missing))}")
    if not days:
        return

    for d in days:
        print(f"\n== {d['day']}: labelled {d['labelled']}/{d['articles']} articles, "
              f"{len(set(d['truth']))} true events, {d['notes']} notes, "
              f"~{d['minutes']:.0f} min active labelling")
        show("metrics (rows = systems)", table(d["truth"], d["systems"]))

    truth = np.concatenate([d["truth"] for d in days])
    systems = {name: np.concatenate([d["systems"][name] for d in days]) for name in days[0]["systems"]}
    groups = np.concatenate([[d["day"]] * len(d["truth"]) for d in days]) if len(days) > 1 else None

    if len(days) > 1:
        show(f"== pooled over {len(days)} days", table(truth, systems))

    baseline = f"replay@{THRESHOLD:.2f}"
    best = max(systems, key=lambda name: summary(truth, systems[name])["bcubed_f1"])
    compared = {name: systems[name] for name in dict.fromkeys(["production", baseline, best])}
    result = bootstrap_bcubed(truth, compared, groups=groups)

    unit = "days" if groups is not None else "true events"
    print(f"\n== B-cubed F1, 95% cluster-bootstrap intervals (resampling {unit})")
    for name, (low, high) in result["f1"].items():
        print(f"  {name:<14} [{low:.3f}, {high:.3f}]")
    for (a, b), (low, high) in result["diff"].items():
        verdict = "differs" if low > 0 or high < 0 else "no clear difference"
        print(f"  {a} - {b}: [{low:+.3f}, {high:+.3f}]  {verdict}")
    if groups is not None and len(days) < 5:
        print("  (only a few days: day-level intervals are rough)")


if __name__ == "__main__":
    main()
