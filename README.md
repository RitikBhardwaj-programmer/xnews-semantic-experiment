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

### GET /health

Returns:

{
  "status": "healthy"
}

## Run locally

```bash
pip install -r requirements.txt

uvicorn app:app --reload