import joblib
import pandas as pd

from pathlib import Path

from feature_extractor import extract_features
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "event_matcher.pkl"
)


# ============================================================
# LOAD EVENT MATCHING MODEL
# ============================================================

model = joblib.load(
    MODEL_PATH
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# CREATE EMBEDDING MAP
# ============================================================

def create_embedding_map(
    article_a_text,
    article_b_text
):

    texts = [
        article_a_text,
        article_b_text
    ]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True
    )

    return {
        article_a_text: embeddings[0],
        article_b_text: embeddings[1]
    }


# ============================================================
# EVENT MATCH PREDICTION
# ============================================================

def predict_event_match(
    article_a_text,
    article_b_text,
    entities_a,
    entities_b,
    date_a,
    date_b,
    location_a,
    location_b
):
    """
    Predict whether two articles belong
    to the same event.

    The function calculates all four
    features before passing them to the
    trained Logistic Regression model.
    """

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    embedding_map = create_embedding_map(
        article_a_text,
        article_b_text
    )


    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    features = extract_features(
        article_a_text,
        article_b_text,
        entities_a,
        entities_b,
        date_a,
        date_b,
        location_a,
        location_b,
        embedding_map
    )


    # --------------------------------------------------------
    # Convert features into DataFrame
    # --------------------------------------------------------

    feature_data = pd.DataFrame(
        [
            {
                "similarity": features[0],
                "entity_score": features[1],
                "temporal_score": features[2],
                "location_score": features[3]
            }
        ]
    )


    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    probability = model.predict_proba(
        feature_data
    )[0][1]

    prediction = model.predict(
        feature_data
    )[0]


    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "probability": float(probability),

        "prediction": (
            "SAME_EVENT"
            if prediction == 1
            else "DIFFERENT_EVENT"
        ),

        "features": {
            "similarity": float(features[0]),
            "entity_score": float(features[1]),
            "temporal_score": float(features[2]),
            "location_score": float(features[3])
        }
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    result = predict_event_match(

        article_a_text=(
            "NovaTech announced its new AI processor "
            "at an event in San Francisco."
        ),

        article_b_text=(
            "NovaTech unveiled a new processor "
            "designed for artificial intelligence "
            "applications in San Francisco."
        ),

        entities_a=[
            "NovaTech",
            "AI processor"
        ],

        entities_b=[
            "NovaTech",
            "processor"
        ],

        date_a="2026-08-10",

        date_b="2026-08-10",

        location_a="San Francisco",

        location_b="San Francisco"
    )


    print("\n========================================")
    print("EVENT MATCH PREDICTION")
    print("========================================")

    print(
        f"Probability: "
        f"{result['probability']:.4f}"
    )

    print(
        f"Prediction : "
        f"{result['prediction']}"
    )

    print("\nFeatures:")

    for name, value in result["features"].items():

        print(
            f"{name:20}: {value:.4f}"
        )