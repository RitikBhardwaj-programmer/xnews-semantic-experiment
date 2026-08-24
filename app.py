from fastapi import FastAPI
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer

from predict import predict_event_match


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="X-NEWS Event Matching Service",
    description="ML service for determining whether two articles describe the same event.",
    version="1.0.0"
)


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

    article_a_text: str
    article_b_text: str

    entities_a: str
    entities_b: str

    date_a: str
    date_b: str

    location_a: str
    location_b: str


# ============================================================
# EVENT MATCH ENDPOINT
# ============================================================

@app.post("/predict")
def predict(request: EventMatchRequest):

    result = predict_event_match(

        article_a_text=request.article_a_text,
        article_b_text=request.article_b_text,

        entities_a=request.entities_a,
        entities_b=request.entities_b,

        date_a=request.date_a,
        date_b=request.date_b,

        location_a=request.location_a,
        location_b=request.location_b
    )

    return result


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