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

### Baseline: 29 Sep 2026 (812 articles, 556 true events)

| System | B-cubed P | B-cubed R | B-cubed F1 | Pairwise P | Pairwise R | CEAF-e F1 | Largest event |
|---|---|---|---|---|---|---|---|
| Production (as it happened) | 0.923 | 0.897 | 0.910 | 0.711 | 0.613 | 0.894 | 10 |
| Replay @ 0.94 | 0.930 | 0.905 | 0.917 | 0.680 | 0.641 | 0.900 | 17 |
| Replay @ 0.96 (best B-cubed F1) | 0.953 | 0.895 | 0.923 | 0.720 | 0.607 | 0.911 | 15 |

- **The threshold alone isn't the fix.** No threshold from 0.90 to 0.99 beats 0.94 clearly: the 95% interval for every difference includes zero. Raising it trades recall for precision almost one for one.
- **The errors go both ways, on the same topics.** In production, 31 events mix two or more real events (117 articles), and 41 real events are split across several production events (175 articles). The worst cases are the India vs West Indies ODIs, the Asian Games and the CEC/SIR political row. Separating events that share a topic needs evidence specific to each event (V4 approach 1), not a different cut-off.
- **Pairwise scores are the stricter view.** B-cubed is inflated by the 451 single-article events, which are easy to get right. Pairwise precision of 0.71 means 29% of the article pairs production puts together are different events.
- **Who labelled it:** day 1 was labelled by Claude at the user's request, not by a person (`labels_2026-09-29.meta.json` records the method). Treat it as provisional until a human spot-check.

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