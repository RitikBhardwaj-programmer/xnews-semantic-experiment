import joblib
import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "event_matcher.pkl"
)

model = joblib.load(MODEL_PATH)


# ============================================================
# ARTICLE-vs-EVENT MATCH PREDICTION
# ============================================================
# The unit being scored is one article against one event centroid,
# not one article against another article. Similarity is computed
# here from the two embeddings so callers can't disagree with the
# model about how it was derived.

def _normalize(vector):

    array = np.asarray(vector, dtype=np.float64)

    norm = np.linalg.norm(array)

    return array if norm == 0 else array / norm


def predict_event_matches(
    article_embedding,
    candidates
):

    if not candidates:
        return []

    article = _normalize(article_embedding)

    similarities = [
        float(
            np.dot(
                article,
                _normalize(candidate["centroid_embedding"])
            )
        )
        for candidate in candidates
    ]

    features = pd.DataFrame({
        "similarity": similarities,
        "temporal_score": [
            candidate["temporal_score"]
            for candidate in candidates
        ]
    })

    probabilities = model.predict_proba(
        features
    )[:, 1]

    return [
        {
            "event_id": candidate["event_id"],
            "probability": float(probability),
            "similarity": similarity
        }
        for candidate, probability, similarity
        in zip(candidates, probabilities, similarities)
    ]


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    rng = np.random.default_rng(0)

    article = rng.normal(size=384)

    results = predict_event_matches(
        article_embedding=article,
        candidates=[
            {
                "event_id": 1,
                "centroid_embedding": article,
                "temporal_score": 1.0
            },
            {
                "event_id": 2,
                "centroid_embedding": rng.normal(size=384),
                "temporal_score": 0.4
            }
        ]
    )

    print("\n========================================")
    print("ARTICLE-vs-EVENT MATCH PREDICTION")
    print("========================================")

    for result in results:
        print(result)
