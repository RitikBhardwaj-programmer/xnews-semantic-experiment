import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load dataset
df = pd.read_csv("data/xnews_batch1_pairs_1-100.csv")

# 2. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Get all unique articles
articles = pd.concat([
    df[["article_a_id", "article_a_text"]].rename(
        columns={"article_a_id": "article_id", "article_a_text": "text"}
    ),
    df[["article_b_id", "article_b_text"]].rename(
        columns={"article_b_id": "article_id", "article_b_text": "text"}
    )
]).drop_duplicates("article_id")

# 4. Generate embeddings
embeddings = model.encode(
    articles["text"].tolist(),
    show_progress_bar=True
)

# 5. Store embedding against article ID
embedding_map = dict(zip(articles["article_id"], embeddings))

# 6. Calculate similarity for every pair
df["similarity"] = df.apply(
    lambda row: cosine_similarity(
        [embedding_map[row["article_a_id"]]],
        [embedding_map[row["article_b_id"]]]
    )[0][0],
    axis=1
)



import re
from datetime import datetime

def temporal_conflict(date_a, date_b, entity_score):
    """
    Measures whether the time difference should strongly
    discourage an event match.

    We only apply the conflict strongly when the articles
    share substantial entity overlap.
    """

    date_a = datetime.strptime(str(date_a), "%Y-%m-%d")
    date_b = datetime.strptime(str(date_b), "%Y-%m-%d")

    days = abs((date_a - date_b).days)

    # No meaningful conflict for very close dates
    if days <= 1:
        return 0.0

    # If entities don't overlap much, don't use this
    # signal aggressively.
    if entity_score < 0.5:
        return 0.0

    if days <= 3:
        return 0.15
    elif days <= 7:
        return 0.30
    elif days <= 14:
        return 0.50
    elif days <= 30:
        return 0.75
    else:
        return 1.0


def normalize_entity(entity):
    """
    Normalize an entity so small formatting differences
    don't prevent matching.
    """
    entity = entity.lower().strip()
    entity = re.sub(r"[^\w\s]", "", entity)
    entity = re.sub(r"\s+", " ", entity)

    return entity


def entity_overlap(entities_a, entities_b):
    """
    Jaccard similarity between the two entity sets.
    """

    set_a = {
        normalize_entity(x)
        for x in str(entities_a).split(";")
        if x.strip()
    }

    set_b = {
        normalize_entity(x)
        for x in str(entities_b).split(";")
        if x.strip()
    }

    if not set_a and not set_b:
        return 0.5

    if not set_a or not set_b:
        return 0.0

    intersection = set_a & set_b
    union = set_a | set_b

    return len(intersection) / len(union)


def temporal_compatibility(date_a, date_b):
    """
    Convert date difference into a compatibility score.

    Same-day articles get 1.0.
    Larger gaps gradually reduce compatibility.
    """

    date_a = datetime.strptime(str(date_a), "%Y-%m-%d")
    date_b = datetime.strptime(str(date_b), "%Y-%m-%d")

    days = abs((date_a - date_b).days)

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


def location_compatibility(location_a, location_b):
    """
    Compare locations.

    Undisclosed locations are treated as neutral
    rather than automatically being considered different.
    """

    a = str(location_a).strip().lower()
    b = str(location_b).strip().lower()

    if a == "undisclosed" or b == "undisclosed":
        return 0.5

    return 1.0 if a == b else 0.0


df["entity_score"] = df.apply(
    lambda row: entity_overlap(
        row["entities_a"],
        row["entities_b"]
    ),
    axis=1
)

df["temporal_score"] = df.apply(
    lambda row: temporal_compatibility(
        row["date_a"],
        row["date_b"]
    ),
    axis=1
)

df["location_score"] = df.apply(
    lambda row: location_compatibility(
        row["location_a"],
        row["location_b"]
    ),
    axis=1
)
df["temporal_conflict"] = df.apply(
    lambda row: temporal_conflict(
        row["date_a"],
        row["date_b"],
        row["entity_score"]
    ),
    axis=1
)



df["hybrid_score"] = (
    0.60 * df["similarity"]
    + 0.20 * df["entity_score"]
    + 0.15 * df["temporal_score"]
    + 0.05 * df["location_score"]
)

df["hybrid_v2_score"] = (
    0.60 * df["similarity"]
    + 0.20 * df["entity_score"]
    + 0.15 * df["temporal_score"]
    + 0.05 * df["location_score"]
    - 0.20 * df["temporal_conflict"]
)

'''
print("\nHybrid results:")

print(
    df[
        [
            "pair_id",
            "ground_truth",
            "hybrid_score",
            "similarity",
            "entity_score",
            "temporal_score",
            "location_score",
            "relationship_type"
        ]
    ].head(30).to_string(index=False)
)
'''




from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Convert ground truth to binary
y_true = (
    df["ground_truth"] == "SAME_EVENT"
).astype(int)

#calculating best hybrid_f1_v1 threshold
# Try a range of thresholds
best_hybrid_threshold = None
best_hybrid_f1 = 0

for threshold in [i / 1000 for i in range(1, 1001)]:

    y_pred = (
        df["hybrid_score"] >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )
    if f1 > best_hybrid_f1:
        best_hybrid_f1 = f1
        best_hybrid_threshold = threshold
        best_hybrid_precision = precision
        best_hybrid_recall = recall

#calculating best hybrid_v2_f1 threshold
best_v2_threshold = None
best_v2_f1 = 0

for threshold in [i / 1000 for i in range(1, 1001)]:

    y_pred = (
        df["hybrid_v2_score"] >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    if f1 > best_v2_f1:
        best_v2_f1 = f1
        best_v2_threshold = threshold
        best_v2_precision = precision
        best_v2_recall = recall

print("\nhybrid:")
print(f"Threshold: {best_hybrid_threshold:.3f}")
print(f"Precision: {best_hybrid_precision:.3f}")
print(f"Recall:    {best_hybrid_recall:.3f}")
print(f"F1:        {best_hybrid_f1:.3f}")

print("\nTemporal-aware hybrid:")
print(f"Threshold: {best_v2_threshold:.3f}")
print(f"Precision: {best_v2_precision:.3f}")
print(f"Recall:    {best_v2_recall:.3f}")
print(f"F1:        {best_v2_f1:.3f}")



'''
# jacard f1
import re

def tokenize(text):
    return set(
        re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    )


def jaccard_similarity(text_a, text_b):
    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)

    if not tokens_a and not tokens_b:
        return 1.0

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b

    return len(intersection) / len(union)

df["jaccard_similarity"] = df.apply(
    lambda row: jaccard_similarity(
        row["article_a_text"],
        row["article_b_text"]
    ),
    axis=1
)

best_jaccard_threshold = None
best_jaccard_f1 = 0

for threshold in [i / 1000 for i in range(1, 1001)]:

    y_pred = (
        df["jaccard_similarity"] >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    if f1 > best_jaccard_f1:
        best_jaccard_f1 = f1
        best_jaccard_threshold = threshold
        best_jaccard_precision = precision
        best_jaccard_recall = recall


'''





# Embedding F1


# Try a range of thresholds
best_threshold = None
best_f1 = 0

for threshold in [i / 1000 for i in range(300, 901)]:

    y_pred = (
        df["similarity"] >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
        best_precision = precision
        best_recall = recall

'''
print("\n========== FINAL COMPARISON ==========")

print(f"Jaccard F1:    {best_jaccard_f1:.3f}")
print(f"Embedding F1: {best_f1:.3f}")
print(f"Hybrid F1:    {best_hybrid_f1:.3f}")
'''
# error matrix

from sklearn.metrics import confusion_matrix

y_pred_hybrid = (
    df["hybrid_score"] >= best_hybrid_threshold
).astype(int)

cm = confusion_matrix(y_true, y_pred_hybrid)

print("\nHybrid confusion matrix:")
print(cm)

print("\nHybrid threshold:")
print(f"{best_hybrid_threshold:.3f}")

print("\nHybrid metrics:")
print(f"Precision: {best_hybrid_precision:.3f}")
print(f"Recall:    {best_hybrid_recall:.3f}")
print(f"F1:        {best_hybrid_f1:.3f}")

errors = df[
    y_true != y_pred_hybrid
].copy()

print("\n========== HYBRID ERRORS ==========")

print(
    errors[
        [
            "pair_id",
            "ground_truth",
            "hybrid_score",
            "similarity",
            "entity_score",
            "temporal_score",
            "location_score",
            "relationship_type",
            "difficulty"
        ]
    ].to_string(index=False)
)

y_pred_v2 = (
    df["hybrid_v2_score"] >= best_v2_threshold
).astype(int)

errors_v2 = df[
    y_true != y_pred_v2
].copy()

print("\n========== V2 ERRORS ==========")

print(
    errors_v2[
        [
            "pair_id",
            "ground_truth",
            "hybrid_v2_score",
            "similarity",
            "entity_score",
            "temporal_score",
            "temporal_conflict",
            "location_score",
            "relationship_type",
            "difficulty"
        ]
    ].to_string(index=False)
)


print("\n========== HYBRID COMPARISON ==========")

print(f"Embedding F1:        {best_f1:.3f}")
print(f"Hybrid V1 F1:        {best_hybrid_f1:.3f}")
print(f"Temporal Hybrid V2:  {best_v2_f1:.3f}")