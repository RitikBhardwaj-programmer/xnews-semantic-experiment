# How X-NEWS evaluates matcher and extractor changes

Lessons from V4 (stages 0–3 and the roadmap extractors, September–October 2026). Read this before running or changing an evaluation. The labelling rules themselves are in `EVENT_DEFINITION.md`.

## Decide before you look
- Write the pass/fail criteria down **before** the run: put them in the script's docstring and print PASS/FAIL. Report the result against them, including failures. Stage 2 (merge pass) and stage 3 (other embedders) were rejected this way; changing the rule after seeing numbers would have shipped noise.
- Prefer the simplest variant that passes (stage 1 chose B over D).
- With three labelled days, use paired cluster-bootstrap intervals (`cluster_metrics.bootstrap_pairwise` / `bootstrap_bcubed`). A gain whose interval includes 0 is not a gain (gte-small: precision 0.821 → 0.895, but pairwise F1 −0.033..+0.061).

## Match production conditions
- **Train the way production will run.** The frozen-vocabulary check "failed" (pairwise F1 −0.048) only because training rows used a vocabulary that contained their own day's words while the test day's didn't. With consistent training (each day's rows use a vocabulary that excludes that day) it passed. `evaluate_stage1.refit_index` does this.
- **Check every feature's range before shipping a model.** `temporal_score` was 1.0 or 0.9 in all 72,151 training rows (the days are consecutive and replays start empty), so the model learned a noise weight that extrapolated to p ≈ 1 for events days old. Compare `StandardScaler.mean_` / `scale_` with what production sends; leave out features without real spread (`evaluate_stage1.SHIPPED_EXCLUDED`).
- Use the stored production embeddings as the baseline, so a re-embedded comparison is against what production actually had (stage 3 reproduced 0.807).
- The service must reproduce the offline features and probabilities exactly: `test_predict_v2.py` checks parity to 1e-9.

## Know what the replay can't show
- `replay.py` starts each day empty, so it never contains multi-day events. `evaluate_stage2.py` replays the three days continuously; production shadow runs are the real test for older events.
- Labels are per day, so scoring is within each day: a correct cross-day continuation isn't rewarded (this works against merge passes and lifetime caps alike).
- **All labels are model-made** (`labels_<day>.meta.json`). Say so in every result, and ask for a human spot-check before relying on small differences.

## Rule-based extractors (entities, claims)
- Run the extractor over all labelled articles and read a sample of its output before opening a PR; tests on hand-picked sentences missed "117-ball" being read as 117 runs.
- Precision first: a reviewer or a user should be able to trust what's shown. Record the extractor version (`extractor_version`) so stored output can be traced and re-run.

## Practicalities
- Use `.venv/Scripts/python.exe`; keep `scikit-learn`/`joblib` at the versions in `requirements.txt` when saving models (the Docker image loads them).
- Folds are independent: run them in parallel with `joblib.Parallel` (`evaluate_stage1.py --refit`: about 2.5 minutes instead of 17).
- Publishers' descriptions and exports stay in git-ignored `local/` folders (`eval_days/local/`, `shadow/local/`); the repository is public.
- Production data is read-only, through the backend repo's `xnews-prod-db` skill, after checking the database host against the Azure secret.
