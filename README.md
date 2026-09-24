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