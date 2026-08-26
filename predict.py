import joblib
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
# EVENT MATCH PREDICTION
# ============================================================

def predict_event_match(
    similarity,
    temporal_score
):

    features = pd.DataFrame([
        {
            "similarity": similarity,
            "temporal_score": temporal_score
        }
    ])

    probability = model.predict_proba(
        features
    )[0][1]

    prediction = model.predict(
        features
    )[0]

    return {
        "probability": float(probability),
        "prediction": (
            "SAME_EVENT"
            if prediction == 1
            else "DIFFERENT_EVENT"
        ),
        "features": {
            "similarity": float(similarity),
            "temporal_score": float(temporal_score)
        }
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    result = predict_event_match(
        similarity=0.80,
        temporal_score=1.0
    )

    print("\n========================================")
    print("EVENT MATCH PREDICTION")
    print("========================================")

    print(
        f"Probability: {result['probability']:.4f}"
    )

    print(
        f"Prediction : {result['prediction']}"
    )

    print(
        f"Similarity : {result['features']['similarity']:.4f}"
    )

    print(
        f"Temporal   : {result['features']['temporal_score']:.4f}"
    )