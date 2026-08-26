from fastapi import FastAPI
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer

from predict import predict_event_match
from datetime import datetime
import numpy as np

# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="X-NEWS Event Matching Service",
    description="ML service for determining whether two articles describe the same event.",
    version="1.0.0"
)

# ============================================================
# SEMANTIC SIMILARITY
# ============================================================

def calculate_similarity(
    text_a: str,
    text_b: str
) -> float:

    embedding_a = embedding_model.encode(
        text_a,
        normalize_embeddings=True
    )

    embedding_b = embedding_model.encode(
        text_b,
        normalize_embeddings=True
    )

    return float(
        np.dot(
            embedding_a,
            embedding_b
        )
    )


# ============================================================
# TEMPORAL COMPATIBILITY
# ============================================================

def calculate_temporal_score(
    date_a: str,
    date_b: str
) -> float:

    parsed_a = datetime.strptime(
        date_a,
        "%Y-%m-%d"
    )

    parsed_b = datetime.strptime(
        date_b,
        "%Y-%m-%d"
    )

    days = abs(
        (parsed_a - parsed_b).days
    )

    if days == 0:
        return 1.0

    elif days <= 1:
        return 0.9

    elif days <= 3:
        return 0.8

    elif days <= 7:
        return 0.6

    elif days <= 30:
        return 0.4

    else:
        return 0.1

# ============================================================
# EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ============================================================
# EVENT MATCH REQUEST
# ============================================================

class EventMatchRequest(BaseModel):

    similarity: float
    temporal_score: float


# ============================================================
# EVENT MATCH ENDPOINT
# ============================================================

@app.post("/predict")
def predict(request: EventMatchRequest):

    return predict_event_match(
        similarity=request.similarity,
        temporal_score=request.temporal_score
    )

# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# EMBEDDING REQUEST
# ============================================================

class EmbeddingRequest(BaseModel):

    text: str


# ============================================================
# EMBEDDING ENDPOINT
# ============================================================

@app.post("/embed")
def generate_embedding(request: EmbeddingRequest):

    embedding = embedding_model.encode(
        request.text,
        normalize_embeddings=True
    )

    return {
        "embedding": embedding.tolist()
    }