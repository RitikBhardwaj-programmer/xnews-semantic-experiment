# X-NEWS AI Event Clustering Microservice

An AI-powered microservice for determining whether two news articles describe the **same real-world event**.

This service is the event-matching component of the X-NEWS system. It uses semantic similarity together with temporal, location, and entity-based signals, and combines them using a trained Logistic Regression classifier.

---

## Overview

News organizations often publish multiple articles about the same real-world event.

For example:

- "Apple unveils a new AI chip"
- "Apple introduces its latest processor for AI workloads"
- "Apple announces new silicon at its California event"

Although these articles have different wording, they may describe the same underlying event.

The purpose of this service is to determine:

```text
Article A
    +
Article B
    ↓
Do they describe the same event?
    ↓
SAME_EVENT / DIFFERENT_EVENT
## Architecture

Article Pair
    ↓
all-MiniLM-L6-v2
    ↓
384-dimensional embeddings
    ↓
Feature Extraction
    ↓
4 Features
    ↓
Logistic Regression
    ↓
Event Match Probability
```
## Features

The deployed model scores one article against one event centroid using two features:

- Semantic similarity (cosine similarity between the article embedding and the event's centroid)
- Temporal compatibility (day-bucket gap between the article's date and the event's last activity)

Entity and location features exist only in the training-time experiments and are not used when serving.

## Model

Embedding model:
sentence-transformers/all-MiniLM-L6-v2

Embedding dimension:
384

Classifier:
Logistic Regression on the two features above, trained on article-vs-event-centroid
rows (10-day event window, top 30 candidates) by:

```bash
python evaluate_centroid_matching.py --write-model models/event_matcher.pkl
```

Training details and the recommended threshold are recorded in
`models/event_matcher.meta.json`. The score is class-balanced and is not a
calibrated probability, so compare it against the threshold configured in the
Java service (`ai.event-matcher.threshold`, currently 0.94) rather than reading
it as a chance of a match.

## Performance

Measured by replaying 1,312 labelled articles as a stream through the same
centroid pipeline the Java service runs (5-fold, split by event):

- Precision ~0.75, recall ~0.89, F1 ~0.81 at threshold 0.94
- Inside the lifecycle window `temporal_score` adds almost nothing beyond
  similarity: a plain cosine threshold of about 0.63 scores within a point of
  the model. Treat the time feature as a minor adjustment.

The dataset is synthetic and deliberately adversarial (pairs that differ by one
word, follow-ups months apart, a few identical texts in different events), so
real-traffic precision is unmeasured. Run `python evaluate_centroid_matching.py`
for the full comparison.

A separate replay of 670 real articles (four Indian outlets, one week) through
the same centroid pipeline showed the difference on real traffic. With the
original pair-trained model at 0.70, the largest event held 121 articles (all
of that week's cricket coverage). With the current model at 0.94 the largest
held 18, all one story, and about 11 of 12 randomly sampled merges were
clearly the same story. There are no labels for that data, and a few
cross-outlet stories about the same event still stay split, which is the cost
of a precision-leaning threshold.

## Evaluation on labelled production days

The figures above come from an *oracle* replay: after every decision the article goes to its true event, so mistakes never pile up. The real grouping quality is measured instead on whole days of production traffic, labelled by hand.

- `data/validation/eval_days/articles_<day>.csv`: every article collected on that UTC day (29 Sep – 1 Oct 2026), with its embedding and the production event it got. RSS descriptions stay local in `eval_days/local/` (git-ignored), because they are publishers' text.
- `data/validation/EVENT_DEFINITION.md`: the labelling rules (one cricket match = one event; a series is a storyline).
- `python label_events.py --day <day>`: a local labelling page on http://localhost:8765. It saves to `labels_<day>.csv` and `events_<day>.csv` after every choice.
- `python evaluate_baseline.py --days <day> [...]`: scores production's real grouping and a free-running replay of the matcher (`replay.py`, which keeps its own mistakes) at thresholds 0.90–0.99. It reports:
  - B-cubed, pairwise and CEAF-e precision, recall and F1, plus ARI
  - the number and size of events
  - 95% cluster-bootstrap intervals
- `python test_cluster_metrics.py`: checks the metrics (`cluster_metrics.py`) on hand-computed cases.

Run these with the project virtual environment (`.venv`).

Sanity check: replaying each day reproduces production's event counts within about 3% (agreement ARI 0.55–0.86). It can't be exact, because production also attaches to events from before the day and processes 3 Kafka partitions in parallel.

### Baseline: 29 Sep – 1 Oct 2026 (2,455 articles, 1,632 true events)

Pooled over the three labelled days (`python evaluate_baseline.py --days 2026-09-29 2026-09-30 2026-10-01`):

| System | B-cubed P | B-cubed R | B-cubed F1 | Pairwise P | Pairwise R | Pairwise F1 | CEAF-e F1 |
|---|---|---|---|---|---|---|---|
| Production (as it happened) | 0.908 | 0.865 | 0.886 | 0.838 | 0.515 | 0.638 | 0.880 |
| Replay @ 0.94 (production rule) | 0.922 | 0.880 | 0.901 | 0.773 | 0.586 | 0.667 | 0.895 |
| Replay @ 0.96 (best B-cubed F1) | 0.950 | 0.868 | 0.907 | 0.890 | 0.524 | 0.659 | 0.899 |

Per day, production's B-cubed F1 was 0.910, 0.873 and 0.874.

- **Splitting is as big a problem as merging.** Pairwise recall of 0.515 means production put only half of the true same-event article pairs together. On 30 Sep, 52 articles about the 2nd ODI ended up in 15 production events, and 25 about the INDIA bloc meeting in 8. On 29 Sep, 31 production events mixed two or more real events. The same topics (cricket, the CEC/SIR row) cause both errors.
- **The threshold alone isn't the fix.** Raising it to 0.96 buys precision with recall, and pairwise F1 barely moves (0.667 → 0.659). Separating and joining coverage correctly needs evidence specific to each event (V4 approach 1): teams and match number, names, time.
- **Production scores below its own replay** (B-cubed F1 difference −0.019 to −0.008, 95% interval over days). This is **not an apples-to-apples comparison.** The replay starts each day with no events, while production also chooses among older, drifted events. Parallel processing in production explains little: only about one production event per large story started within two minutes of another. With three days, day-level intervals are rough.
- **B-cubed is generous here.** About 60% of true events are single articles, which are easy to get right, so pairwise and CEAF-e are the stricter views.
- **Who labelled them:** all three days were labelled by Claude at the user's request, not by a person (`labels_<day>.meta.json`). Treat the labels as provisional until a human spot-check. On 1 Oct, the 68-article flydubai event carries one note on all its rows, which inflates that day's note count.

## Stage 1: event-evidence matcher (offline)

`python evaluate_stage1.py` (about 17 minutes) adds evidence to the attach decision (`event_features.py`):
- **member similarities:** max, min, top-3 and newest member
- **TF-IDF wording:** cosine of template-stripped titles, and of title+description
- **entities and numbers**
- **a cricket fixture signature**, used as a veto
- **time and size**

It tests these with **leave-one-day-out**: train on two labelled days (oracle replay), choose the threshold on those days only, then replay the third day free-running. TF-IDF is fitted on all three days' text; no labels are used.

| Variant (pooled, 3 held-out days) | B-cubed F1 | Pairwise P | Pairwise R | Pairwise F1 | Mixed events | 2nd ODI split into |
|---|---|---|---|---|---|---|
| replay@0.94 (production rule) | 0.901 | 0.773 | 0.586 | 0.667 | 90 | 6 |
| production rule + hard member gate ≥ 0.45 | 0.898 | 0.882 | 0.443 | 0.590 | 89 | 12 |
| A: + member similarities | 0.902 | 0.649 | 0.641 | 0.645 | 60 | 6 |
| **B: A + TF-IDF wording** | **0.917** | **0.814** | **0.754** | **0.783** | **90** | **5** |
| C: B + entities, numbers | 0.914 | 0.757 | 0.790 | 0.773 | 89 | 3 |
| D: C + fixture veto | 0.910 | 0.809 | 0.726 | 0.765 | 89 | 5 |
| E: D + time, size (all features) | 0.913 | 0.816 | 0.674 | 0.738 | 74 | 6 |
| F: E, re-embedding stripped titles | 0.915 | 0.851 | 0.669 | 0.749 | 71 | 6 |
| G: E with gradient boosting | 0.913 | 0.851 | 0.671 | 0.750 | 66 | 6 |
| H: E + hard member gate | 0.902 | 0.911 | 0.415 | 0.570 | 69 | 15 |

**Decision, by the criteria fixed before the run:** pairwise F1 gain with a 95% interval above 0, B-cubed F1 not worse, mixed events not more, and the 2nd ODI coverage split less. **B passes:** pairwise F1 +0.116, interval +0.045 to +0.181; B-cubed F1 +0.016, interval +0.005 to +0.030. D also passes, so the simpler B is chosen.

- **Wording is the big win.** TF-IDF on stripped titles and descriptions joins coverage of one story that the embedding alone keeps apart. On 30 Sep, pairwise recall rose from 0.48 to 0.67.
- **The hard member gate is rejected.** It raises precision but splits coverage badly. On 30 Sep the 2nd ODI ended up in 12–15 events, and pairwise recall fell to 0.22.
- **E, F and G give fewer mixed events (66–74 vs 90)** but fail the split criterion by a tie (6 vs 6). They are worth revisiting if over-merging matters more than splitting.
- **The fixture veto helps exactly where expected:** on 1 Oct, pairwise precision rose from 0.868 to 0.930 (C → D).
- **Caveats:**
  - **Uneven across days:** B helps mostly on 30 Sep and 1 Oct. On 29 Sep its pairwise F1 matches the baseline (0.657 vs 0.660), and it puts 10% more articles into mixed events.
  - **Small, model-labelled data:** three days, labelled by a model. The pairwise intervals are wide, because a few large events dominate pair counts.

### Stage 1b step 0: variant B under production conditions

Production can't see a day's words before it happens, and it sends only a bounded number of member texts. Two flags test B under those conditions on the same folds:
- `python evaluate_stage1.py --checks` (about 17 minutes) runs a frozen vocabulary and a 20-member text cap.
- `python evaluate_stage1.py --refit` (about 3 minutes, folds in parallel) runs a vocabulary refit.

| Variant B configuration (pooled) | B-cubed F1 | Pairwise P | Pairwise R | Pairwise F1 | Verdict |
|---|---|---|---|---|---|
| as in stage 1 (TF-IDF fitted on all days) | 0.917 | 0.814 | 0.754 | 0.783 | reference |
| 20-member text cap | 0.915 | 0.800 | 0.783 | 0.791 | pass |
| frozen vocabulary (training days only) | 0.890 | 0.644 | 0.855 | 0.734 | **fail** (F1 −0.048) |
| frozen vocabulary + cap 20 | 0.896 | 0.721 | 0.836 | 0.774 | F1 pass, precision −0.093 |
| **nightly refit + cap 20, consistent training** | **0.917** | **0.804** | **0.796** | **0.800** | **recovers** |
| hourly refit + cap 20, consistent training | 0.917 | 0.820 | 0.732 | 0.773 | recovers |
| **nightly refit + cap 20, no `temporal_score` (shipped)** | **0.920** | **0.821** | **0.794** | **0.807** | **recovers** |

Pass rule: pairwise F1 within −0.02 of B. For a refit to count as recovering, pair precision must also stay within −0.03 of B; this rule was set before the refit run.

- **The frozen-vocabulary failure is mostly a train/test mismatch.** In that check, training rows used a vocabulary that contained their own day's words, and the test day's did not. With *consistent training* (every day's rows use a vocabulary that excludes that day), precision comes back.
- **Nightly is chosen over hourly.** It is better on F1 and simpler to run. The shipped matcher refits its vocabulary nightly on the previous 14 days.
- **`temporal_score` is left out of the shipped model.** In every training row it is 1.0 or 0.9, because the labelled days are consecutive and each replay starts empty. The model therefore cannot learn it, and it gave a small negative weight that extrapolates to near-certain matches below 0.9. A local shadow run caught this: events several days old (temporal 0.4 or 0.1) were joined to unrelated articles with p ≈ 0.9999. Without the feature, the offline result is slightly better. The service still records the value with each decision.
- **Not covered offline:** multi-day events. Each replay starts empty, so the shadow run on production traffic is the first test with events up to 10 days old.
- **Caveat:** with only three days, each simulated vocabulary came from the neighbouring days, sometimes later ones. Labels are model-made.

### Stage 1b step 3: shadow run in production

Production has run with `AI_EVENT_MATCHER_MODE=shadow` since 2026-10-02 07:43 UTC. v1 decides, and both decisions are stored in `event_match_decisions`. After 3 days:

1. `python evaluate_shadow.py export-sql --from "2026-10-02 07:43" --to "2026-10-05 07:45"` writes a read-only `\copy` script, which you run with psql from `data/validation/shadow/local/` (git-ignored, because it contains descriptions).
2. `python evaluate_shadow.py evaluate` reports:
   - v2 health: errors by kind, and p50/p95 latency
   - v1/v2 agreement
   - `disagreements.csv` for labelling against `EVENT_DEFINITION.md`. Fill in `label` with `v1`, `v2`, `neither` or `unsure`, then save it as `disagreements_labelled.csv`.
   - a free-running v2 replay of the whole window. It runs continuously, so events can span days, and it refits the vocabulary nightly as production does.

   It ends with PASS/FAIL against the go-live criteria fixed in the stage 1b plan:
   - v2's share of the disagreements it gets right has a 95% Wilson interval above 50%
   - v2 error rate < 1%
   - p95 latency < 1.5 s
   - no replay event larger than 2× the largest labelled true event (68, so the limit is 136)

## Stage 2: lifetime cap and nightly merge pass (offline, not adopted)

`python evaluate_stage2.py` (about 5 minutes) replays the three labelled days **continuously**, so events can span days, with the shipped v2 setup (variant B without `temporal_score`, nightly-style vocabulary, consistent training, 20-member cap, leave-one-day-out models). It tests:
- a maximum event lifetime: no new articles join an event whose first activity is more than k days old
- a merge-only pass at each day boundary: an event joins a larger active event (last 72 h, top 5 by centroid cosine) if its members' mean v2 probability reaches the model threshold

Scoring is within each day, because the labels are per day. "Mixed" counts predicted events that, within one day, contain articles of two or more true events.

| Setup (pooled) | B-cubed F1 | Pairwise P | Pairwise R | Pairwise F1 | Events | Mixed |
|---|---|---|---|---|---|---|
| v2 continuous | 0.908 | 0.738 | 0.795 | 0.765 | 1,559 | 104 |
| + lifetime cap 2 days | 0.907 | 0.746 | 0.755 | 0.751 | 1,588 | 97 |
| + lifetime cap 3 days | 0.908 | 0.738 | 0.795 | 0.765 | 1,559 | 104 |
| + merge pass | 0.893 | 0.691 | 0.923 | 0.790 | 1,414 | 131 |

**Decision, by the criteria fixed before the run:**
- **Merge pass: fails.** It raises recall, but B-cubed F1 drops (difference −0.024 to −0.004, 95% interval), mixed events rise from 104 to 131, and the pairwise F1 gain isn't significant (−0.041 to +0.079). Not adopted.
- **Lifetime cap: not adopted.**
  - 3 days is identical to no cap; three days of data can't test it.
  - 2 days meets the lenient "no harm" rule (intervals include 0, 7 fewer mixed events), but lowers mean pairwise F1 by 0.014 and recall by 0.04.
  - Production already closes events after 10 days of inactivity.
- **Caveats:** three model-labelled days; per-day labels can't reward correct cross-day continuation, which works against the merge pass and the cap alike.

## API

### POST /predict

Scores one new article against a list of candidate **event centroids** in a
single call and returns a same-event probability for each. Requires an
`X-API-Key` header.

#### Request

```json
{
  "article_embedding": [0.01, "... 384 floats ..."],
  "candidates": [
    { "event_id": 12, "centroid_embedding": ["... 384 floats ..."], "temporal_score": 0.9 },
    { "event_id": 42, "centroid_embedding": ["... 384 floats ..."], "temporal_score": 0.4 }
  ]
}
```

Similarity is computed server-side (cosine) from the two embeddings.

#### Response
```json
{
  "results": [
    { "event_id": 12, "probability": 0.91, "similarity": 0.83 },
    { "event_id": 42, "probability": 0.37, "similarity": 0.61 }
  ]
}
```

### POST /predict/v2

The variant B matcher (stage 1b), alongside `/predict` (which is unchanged). It needs an `X-API-Key` header. The backend computes the centroid similarity and the member similarities in SQL over all members, and sends the texts of up to 20 newest members, newest first. The service adds the TF-IDF wording features (`event_features.py`, the same code as offline) and scores with `models/event_matcher_v2.joblib`. The attach decision (best probability ≥ `threshold`) stays in the backend.

#### Request
```json
{
  "title": "IND vs WI, 2nd ODI LIVE score: Kohli in action",
  "description": "India look to seal the series ...",
  "candidates": [
    { "event_id": 12, "similarity": 0.91, "temporal_score": 1.0,
      "member_max": 0.93, "member_min": 0.71, "member_top3": 0.9, "member_newest": 0.88,
      "members": [ { "title": "India vs West Indies second ODI ...", "description": "..." } ] }
  ]
}
```

Limits (422 otherwise): at most 30 candidates; 1–20 members per candidate; titles of 1–500 characters; descriptions of at most 5,000 characters; similarities in [−1, 1]; `temporal_score` in [0, 1].

#### Response
```json
{
  "model_version": "v2-b-notemporal-2026-10-02",
  "vocabulary_version": "2026-10-03T02:00:04Z/11480",
  "threshold": 0.99,
  "results": [
    { "event_id": 12, "probability": 0.993, "features": { "similarity": 0.91, "...": "8 values" } }
  ]
}
```

`vocabulary_version` is `"default"` until the first refit after the service starts.

### POST /vocabulary/v2

Refits the TF-IDF vocabulary used by `/predict/v2`. It needs an `X-API-Key` header. The backend sends the title and description of recent articles nightly (the last 14 days), and again whenever `/predict/v2` reports `"default"`. The request body is `{"items": [{"title": "...", "description": "..."}]}`, with 100–20,000 items. The response is `{"vocabulary_version": "...", "documents": n}`. The new vocabulary is swapped in as a whole, and if fitting fails (422) the old one stays. It is kept in memory only, so after a restart the shipped default vocabulary is used. At 12,000 items (3 MB) a refit takes about 1.5 s locally.

Train the model with `python train_event_matcher_v2.py`, which needs the local descriptions. Each day's training rows use a vocabulary fitted on the other days, matching the nightly refit; see stage 1b step 0. `python test_predict_v2.py` checks that the service reproduces the offline features and probabilities to 1e-9.

### POST /entities

Entity mentions in one article, for the information layer (V4 roadmap step 3). It needs an `X-API-Key` header. The request is `{"title": "...", "description": "..."}`, with the same limits as `ArticleText`.

The response is `{"extractor_version": "rules-2026-10-02", "mentions": [{"text": "Mulla Afroz", "normalized": "mulla afroz", "type": "name", "field": "title"}]}`. There is one mention per entity and field. `type` is `team` (from the gazetteer in `event_features.py`) or `name` (a capitalised run: a person, place or organisation). The backend stores the mentions and resolves aliases.

`entity_extraction.py` uses rules on purpose, so there is no new dependency:
- names never run across punctuation
- title-case headlines are skipped, so only their description is used
- format words, weekdays and months are dropped
- names are at most 5 words long
- a team at the start of a longer name ("India Meteorological Department") is not a team

On the 2,455 labelled articles it finds 3.2 entities per article on average (187 articles have none). Expect some noise: demonyms ("Indian") and job titles kept with names ("captain Smit Machchhar"). `python test_entity_extraction.py` checks it.

### POST /claims

Numeric cricket claims in one article, for V4 roadmap step 6. It needs an `X-API-Key` header. The request is `{"title": "...", "description": "..."}`.

The response is `{"extractor_version": "cricket-rules-2026-10-02", "claims": [{"subject": "Virat Kohli", "subject_normalized": "virat kohli", "predicate": "runs_scored", "value": 139, "value_text": "139", "unit": "runs", "quote": "Virat Kohli's unbeaten 139", "start": 12, "end": 38, "field": "description"}]}`.

Predicates: `innings_score` (351/9), `all_out_for`, `chased`, `set_target`, `runs_scored`, `bowling_figures` (6/75), `won_by` and `lost_by`.

`claim_extraction.py` uses rules and puts precision first:
- the article must look like cricket
- the subject must be a name right next to the number pattern
- role words are stripped, and a subject starting with a team is cut to the team
- a team followed by "'s" and a number is a total, not a player's runs
- wickets are at most 10

On the 2,455 labelled articles it finds 32 claims (23 distinct). They look plausible on manual inspection; one false reading ("117-ball" as runs) was found in local review and fixed. The backend shows a claim only after an admin approves it. `python test_claim_extraction.py` checks it.

### GET /health

Returns:

{
  "status": "healthy"
}

## Run locally

```bash
pip install -r requirements.txt

uvicorn app:app --reload