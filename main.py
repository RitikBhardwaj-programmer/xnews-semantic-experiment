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

# 7. Display results
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


'''
print(
    df[
        [
            "pair_id",
            "ground_truth",
            "similarity",
            "difficulty",
            "relationship_type"
        ]
    ].head(100).to_string(index=False)
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


print("\nBest threshold:")
print(f"Threshold: {best_threshold:.3f}")
print(f"Precision: {best_precision:.3f}")
print(f"Recall: {best_recall:.3f}")
print(f"F1: {best_f1:.3f}")



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

print("\nJaccard baseline:")
print(f"Threshold: {best_jaccard_threshold:.3f}")
print(f"Precision: {best_jaccard_precision:.3f}")
print(f"Recall: {best_jaccard_recall:.3f}")
print(f"F1: {best_jaccard_f1:.3f}")



print("\nFinal comparison:")
print("--------------------------------")
print(f"Jaccard F1:    {best_jaccard_f1:.3f}")
print(f"Embedding F1: {best_f1:.3f}")