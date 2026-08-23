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
BGE-base-en-v1.5
    ↓
768-dimensional embeddings
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

Determines whether two news articles describe the same real-world event.

#### Request

```json
{
  "article_a_text": "NovaTech announced its new AI processor in San Francisco.",
  "article_b_text": "NovaTech unveiled a new processor designed for AI applications in San Francisco.",
  "entities_a": "NovaTech;AI processor",
  "entities_b": "NovaTech;processor",
  "date_a": "2026-08-10",
  "date_b": "2026-08-10",
  "location_a": "San Francisco",
  "location_b": "San Francisco"
}
```

#### Response
```json
{
  "probability": 0.9840580487944166,
  "prediction": "SAME_EVENT",
  "features": {
    "similarity": 0.944098711013794,
    "entity_score": 0.3333333333333333,
    "temporal_score": 1,
    "location_score": 1
  }
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