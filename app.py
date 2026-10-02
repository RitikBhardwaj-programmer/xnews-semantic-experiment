import os

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from sentence_transformers import SentenceTransformer

from matcher_v2 import MEMBER_TEXT_CAP, load_matcher
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
# EVENT MATCH V2 (variant B, stage 1b)
# ============================================================
# The backend computes the centroid and member similarities in SQL and
# sends the newest members' texts; the service adds the TF-IDF wording
# features and scores. The attach decision (best probability >= threshold)
# stays in the backend.

event_matcher_v2 = load_matcher()

MAX_CANDIDATES = 30
MAX_TITLE_CHARS = 500
MAX_DESCRIPTION_CHARS = 5000
MIN_VOCABULARY_ITEMS = 100
MAX_VOCABULARY_ITEMS = 20000


class ArticleText(BaseModel):

    title: str = Field(min_length=1, max_length=MAX_TITLE_CHARS)
    description: str | None = Field(default=None, max_length=MAX_DESCRIPTION_CHARS)


class EventCandidateV2(BaseModel):

    event_id: int
    similarity: float = Field(ge=-1, le=1)
    temporal_score: float = Field(ge=0, le=1)
    member_max: float = Field(ge=-1, le=1)
    member_min: float = Field(ge=-1, le=1)
    member_top3: float = Field(ge=-1, le=1)
    member_newest: float = Field(ge=-1, le=1)
    members: list[ArticleText] = Field(min_length=1, max_length=MEMBER_TEXT_CAP)


class EventMatchRequestV2(BaseModel):

    title: str = Field(min_length=1, max_length=MAX_TITLE_CHARS)
    description: str | None = Field(default=None, max_length=MAX_DESCRIPTION_CHARS)
    candidates: list[EventCandidateV2] = Field(max_length=MAX_CANDIDATES)


@app.post("/predict/v2", dependencies=[Depends(require_api_key)])
def predict_v2(request: EventMatchRequestV2):

    return event_matcher_v2.predict(
        title=request.title,
        description=request.description,
        candidates=[
            candidate.model_dump()
            for candidate in request.candidates
        ]
    )


class VocabularyRequest(BaseModel):

    items: list[ArticleText] = Field(
        min_length=MIN_VOCABULARY_ITEMS,
        max_length=MAX_VOCABULARY_ITEMS
    )


@app.post("/vocabulary/v2", dependencies=[Depends(require_api_key)])
def refit_vocabulary_v2(request: VocabularyRequest):

    items = [item.model_dump() for item in request.items]

    try:
        version = event_matcher_v2.refit(items)
    except ValueError as error:
        # e.g. only stop words: keep the current vocabulary
        raise HTTPException(status_code=422, detail=f"Vocabulary not refitted: {error}")

    return {
        "vocabulary_version": version,
        "documents": len(items)
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