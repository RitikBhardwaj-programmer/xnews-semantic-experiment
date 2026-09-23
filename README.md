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

The model uses:

- Semantic similarity
- Entity overlap
- Temporal compatibility
- Location compatibility

## Model

Embedding model:
sentence-transformers/all-MiniLM-L6-v2

Embedding dimension:
384


Classifier:
Logistic Regression

## Performance

Dataset:
900 article pairs

Accuracy:
~89.7%

F1:
~89.5%

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