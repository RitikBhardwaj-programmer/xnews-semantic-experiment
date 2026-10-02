"""
Train the v2 event matcher (variant B, V4 approach 1 stage 1b) and save it
for /predict/v2.

- Training rows: oracle replay of all labelled days. Each day's TF-IDF
  features use a vocabulary fitted on the OTHER days only and the 20-member
  text cap ("consistent training"), because production scores every article
  with a vocabulary refitted the night before, which never contains that
  day's own words. Stage 1b step 0 showed this is what keeps precision.
- Model and threshold: the variant B logistic regression and threshold rule
  from evaluate_stage1.py, fitted on all days' rows, without temporal_score
  (see evaluate_stage1.SHIPPED_EXCLUDED: the labelled days can't teach it).
- Default vocabulary: fitted on all labelled days with the same code as
  POST /vocabulary/v2; used until the backend's first refit.

Needs the local descriptions (data/validation/eval_days/local/).

Usage:
    python train_event_matcher_v2.py
"""

import json
from datetime import datetime, timezone

import joblib
import pandas as pd
import sklearn

from evaluate_stage1 import DAYS, VARIANTS, choose_threshold, fit, load, oracle_rows, refit_index, shipped_columns
from matcher_v2 import MEMBER_TEXT_CAP, META_PATH, MODEL_PATH, fit_vocabulary

VERSION = "v2-b-notemporal-2026-10-02"


def main():
    name, groups, veto, gate, kind, _ = VARIANTS[1]
    columns = shipped_columns()
    frames = [load(d) for d in DAYS]
    days = range(len(DAYS))

    rows = pd.concat([
        oracle_rows(frames[d], refit_index(frames, d, [c for c in days if c != d], MEMBER_TEXT_CAP, hourly=False))
        for d in days
    ])
    model = fit(rows, columns, kind)
    threshold = float(choose_threshold(model, rows, columns, veto))

    allf = pd.concat(frames)
    items = [{"title": t, "description": d if d == d else None} for t, d in zip(allf["title"], allf["description"])]
    title_vectorizer, text_vectorizer = fit_vocabulary(items)

    joblib.dump({
        "version": VERSION,
        "model": model,
        "columns": columns,
        "threshold": threshold,
        "title_vectorizer": title_vectorizer,
        "text_vectorizer": text_vectorizer,
    }, MODEL_PATH)

    meta = {
        "version": VERSION,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_file": "train_event_matcher_v2.py",
        "variant": name,
        "trained_on": f"oracle replay of labelled days {', '.join(DAYS)} (model-made labels), "
                      f"each day's TF-IDF fitted on the other days, member text cap {MEMBER_TEXT_CAP}",
        "articles": int(len(allf)),
        "true_events": int(sum(f["truth"].nunique() for f in frames)),
        "training_rows": int(len(rows)),
        "attach_rows": int(rows["is_true"].sum()),
        "features": columns,
        "coefficients": dict(zip(columns, [round(float(c), 3) for c in model[-1].coef_[0]])),
        "threshold": threshold,
        "default_vocabulary": {"documents": int(len(items)),
                               "title_terms": len(title_vectorizer.vocabulary_),
                               "text_terms": len(text_vectorizer.vocabulary_)},
        "offline_evaluation": "README.md, Stage 1b step 0: nightly refit + cap 20, leave-one-day-out, "
                              "no temporal: pooled pairwise P 0.821 / R 0.794 / F1 0.807 vs production rule F1 0.667",
        "scikit_learn": sklearn.__version__,
    }
    META_PATH.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
