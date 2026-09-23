import os

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from sentence_transformers import SentenceTransformer

from predict import predict_event_matches
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
# API KEY AUTH
# ============================================================
# The Java backend's JWT layer does not protect this service directly -
# it is reachable at whatever URL Azure Container Apps exposes it on.
# This header check is the minimal gate so /embed and /predict aren't
# open to anyone who finds the URL.

API_KEY = os.environ["AI_SERVICE_API_KEY"]


def require_api_key(x_api_key: str = Header(...)):

    if x_api_key != API_KEY:

        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
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

EMBEDDING_DIMENSIONS = 384


class EventCandidate(BaseModel):

    event_id: int
    centroid_embedding: list[float] = Field(
        min_length=EMBEDDING_DIMENSIONS,
        max_length=EMBEDDING_DIMENSIONS
    )
    temporal_score: float


class EventMatchRequest(BaseModel):

    article_embedding: list[float] = Field(
        min_length=EMBEDDING_DIMENSIONS,
        max_length=EMBEDDING_DIMENSIONS
    )
    candidates: list[EventCandidate]


# ============================================================
# EVENT MATCH ENDPOINT
# ============================================================
# One article scored against many event centroids in a single call,
# instead of one round trip per candidate.

@app.post("/predict", dependencies=[Depends(require_api_key)])
def predict(request: EventMatchRequest):

    return {
        "results": predict_event_matches(
            article_embedding=request.article_embedding,
            candidates=[
                candidate.model_dump()
                for candidate in request.candidates
            ]
        )
    }

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

@app.post("/embed", dependencies=[Depends(require_api_key)])
def generate_embedding(request: EmbeddingRequest):

    embedding = embedding_model.encode(
        request.text,
        normalize_embeddings=True
    )

    return {
        "embedding": embedding.tolist()
    }