import re
import numpy as np
import pandas as pd
import joblib
from feature_extractor import extract_features, location_compatibility, temporal_compatibility, entity_overlap, \
    semantic_similarity
from sklearn.model_selection import GroupShuffleSplit
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. LOAD ALL X-NEWS CSV FILES
# ============================================================

from pathlib import Path

DATA_DIR = Path("data")

# Find all CSV files inside data/
DATA_FILES = sorted(DATA_DIR.glob("*.csv"))

print("CSV files found:")

for file in DATA_FILES:
    print(" -", file)

if not DATA_FILES:
    raise FileNotFoundError(
        "No CSV files found inside the data/ directory."
    )


# Load each batch
dataframes = []

for file in DATA_FILES:

    batch = pd.read_csv(file)

    print(
        f"{file.name}: {len(batch)} pairs"
    )

    dataframes.append(batch)


# Combine all batches
df = pd.concat(
    dataframes,
    ignore_index=True
)


print("\n========================================")
print("DATASET")
print("========================================")

print(
    "Total article pairs:",
    len(df)
)

print(
    "Columns:",
    list(df.columns)
)

# ============================================================
# 2. LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print(
    "Embedding dimension:",
    model.get_embedding_dimension()
)


# ============================================================
# 3. GENERATE EMBEDDINGS
# ============================================================

print("\nGenerating embeddings...")

all_articles = (
    pd.concat([
        df["article_a_text"],
        df["article_b_text"]
    ])
    .fillna("")
    .astype(str)
    .unique()
    .tolist()
)

embeddings = model.encode(
    all_articles,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

embedding_map = dict(
    zip(all_articles, embeddings)
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

print("\nExtracting features...")

feature_rows = []

for _, row in df.iterrows():

    features = extract_features(
        row["article_a_text"],
        row["article_b_text"],
        row["entities_a"],
        row["entities_b"],
        row["date_a"],
        row["date_b"],
        row["location_a"],
        row["location_b"],
        embedding_map
    )

    feature_rows.append(features)


df[
    [
        "similarity",
        "entity_score",
        "temporal_score",
        "location_score"
    ]
] = feature_rows

print("Features extracted successfully.")


# ============================================================
# 9. CREATE TARGET
# ============================================================

y = (
    df["ground_truth"] == "SAME_EVENT"
).astype(int)


# ============================================================
# 10. SELECT FEATURES
# ============================================================

features = [
    "similarity",
    "entity_score",
    "temporal_score",
    "location_score"
]

X = df[features]

# ============================================================
# EVENT-AWARE GROUPING
# ============================================================

class UnionFind:

    def __init__(self):
        self.parent = {}

    def find(self, x):

        if x not in self.parent:
            self.parent[x] = x

        if self.parent[x] != x:
            self.parent[x] = self.find(
                self.parent[x]
            )

        return self.parent[x]

    def union(self, a, b):

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a != root_b:
            self.parent[root_b] = root_a


uf = UnionFind()


# Connect events that appear together in a pair
for _, row in df.iterrows():

    event_a = str(row["event_a_id"])
    event_b = str(row["event_b_id"])

    uf.union(event_a, event_b)


# Assign each pair to an event component
groups = []

for _, row in df.iterrows():

    event_a = str(row["event_a_id"])

    groups.append(
        uf.find(event_a)
    )


groups = np.array(groups)


print("\n========================================")
print("EVENT-AWARE GROUPING")
print("========================================")

print(
    "Unique event groups:",
    len(np.unique(groups))
)
print(
    "\nGroup sizes:"
)

print(
    pd.Series(groups).value_counts().describe()
)


# ============================================================
# EVENT-AWARE TRAIN / TEST SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
)


X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]


print("\nDataset split:")

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples :",
    len(X_test)
)


print("\nTraining class distribution:")

print(
    y_train.value_counts()
)

print("\nTesting class distribution:")

print(
    y_test.value_counts()
)


# ============================================================
# 12. TRAIN LOGISTIC REGRESSION
# ============================================================

print("\nTraining Logistic Regression...")

classifier = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

classifier.fit(
    X_train,
    y_train
)
'''
# ============================================================
# SAVE TRAINED EVENT MATCHER
# ============================================================

from pathlib import Path

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "event_matcher.pkl"

joblib.dump(
    classifier,
    MODEL_PATH
)

print("\n========================================")
print("MODEL SAVED")
print("========================================")

print(
    "Model path:",
    MODEL_PATH
)


# ============================================================
# TEST SAVED MODEL
# ============================================================

loaded_model = joblib.load(
    MODEL_PATH
)

sample_features = pd.DataFrame([
    {
        "similarity": 0.80,
        "entity_score": 0.67,
        "temporal_score": 1.0,
        "location_score": 1.0
    }
])

probability = loaded_model.predict_proba(
    sample_features
)[0][1]

prediction = loaded_model.predict(
    sample_features
)[0]

print("\n========================================")
print("SAVED MODEL TEST")
print("========================================")

print(
    f"Probability of SAME_EVENT: {probability:.4f}"
)

print(
    "Prediction:",
    "SAME_EVENT" if prediction == 1
    else "DIFFERENT_EVENT"
)
'''
# ============================================================
# 13. PREDICTIONS
# ============================================================

y_pred = classifier.predict(
    X_test
)

y_probability = classifier.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 14. EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print("\n========================================")
print("       LOGISTIC REGRESSION RESULTS")
print("========================================")

print(
    f"Accuracy : {accuracy:.3f}"
)

print(
    f"Precision: {precision:.3f}"
)

print(
    f"Recall   : {recall:.3f}"
)

print(
    f"F1       : {f1:.3f}"
)


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 16. DETAILED CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "DIFFERENT_EVENT",
            "SAME_EVENT"
        ],
        zero_division=0
    )
)


# ============================================================
# LEARNED FEATURE COEFFICIENTS
# ============================================================

print("\n========================================")
print("       LEARNED FEATURE COEFFICIENTS")
print("========================================")

# Get the Logistic Regression model
# from inside the Pipeline
logistic_model = classifier.named_steps["classifier"]

for feature, coefficient in zip(
    features,
    logistic_model.coef_[0]
):
    print(
        f"{feature:20} {coefficient:+.4f}"
    )

print(
    f"\nIntercept: {logistic_model.intercept_[0]:+.4f}"
)

# ============================================================
# 18. INTERCEPT
# ============================================================

print(
    f"\nIntercept: {logistic_model.intercept_[0]:+.4f}"
)


# ============================================================
# 19. SAMPLE PREDICTIONS
# ============================================================

test_results = X_test.copy()

test_results["actual"] = y_test.values

test_results["predicted"] = y_pred

test_results["probability_same_event"] = (
    y_probability
)

print("\n========================================")
print("          SAMPLE PREDICTIONS")
print("========================================")

print(
    test_results.head(20).to_string()
)


# ============================================================
# 20. FINISHED
# ============================================================

print("\nProcess finished successfully.")