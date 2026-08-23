import re
from datetime import datetime
import numpy as np


# ============================================================
# ENTITY NORMALIZATION
# ============================================================

def normalize_entity(entity):
    """
    Basic normalization of an entity.

    Example:
        " Elon Musk " -> "elon musk"
        "Elon-Musk!" -> "elonmusk"
    """

    entity = str(entity).strip().lower()

    entity = re.sub(
        r"[^\w\s]",
        "",
        entity
    )

    entity = re.sub(
        r"\s+",
        " ",
        entity
    ).strip()

    return entity


# ============================================================
# ENTITY OVERLAP
# ============================================================

def entity_overlap(entities_a, entities_b):
    """
    Calculates Jaccard similarity between the
    entities of two articles.
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

    # Both articles have no entities.
    # Treat this as neutral rather than identical.
    if not set_a and not set_b:
        return 0.5

    # One has entities and the other doesn't.
    if not set_a or not set_b:
        return 0.0

    intersection = set_a & set_b
    union = set_a | set_b

    return len(intersection) / len(union)


# ============================================================
# TEMPORAL COMPATIBILITY
# ============================================================

def temporal_compatibility(date_a, date_b):
    """
    Calculates compatibility based on the difference
    between article publication dates.
    """

    date_a = datetime.strptime(
        str(date_a),
        "%Y-%m-%d"
    )

    date_b = datetime.strptime(
        str(date_b),
        "%Y-%m-%d"
    )

    days = abs(
        (date_a - date_b).days
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
# LOCATION COMPATIBILITY
# ============================================================

def location_compatibility(location_a, location_b):
    """
    Calculates compatibility between article locations.
    """

    a = str(location_a).strip().lower()
    b = str(location_b).strip().lower()

    # Unknown location is neutral.
    if (
        a == "undisclosed"
        or b == "undisclosed"
    ):
        return 0.5

    if a == b:
        return 1.0

    return 0.0


# ============================================================
# COSINE / SEMANTIC SIMILARITY
# ============================================================

def semantic_similarity(
    text_a,
    text_b,
    embedding_map
):
    """
    Calculates cosine similarity between two
    precomputed article embeddings.

    Because embeddings are normalized,
    cosine similarity = dot product.
    """

    embedding_a = embedding_map[text_a]
    embedding_b = embedding_map[text_b]

    return float(
        np.dot(
            embedding_a,
            embedding_b
        )
    )


# ============================================================
# EXTRACT ALL FEATURES
# ============================================================

def extract_features(
    article_a_text,
    article_b_text,
    entities_a,
    entities_b,
    date_a,
    date_b,
    location_a,
    location_b,
    embedding_map
):
    """
    Extract the four features used by the
    event-matching model.

    Returns:
        [similarity,
         entity_score,
         temporal_score,
         location_score]
    """

    similarity = semantic_similarity(
        article_a_text,
        article_b_text,
        embedding_map
    )

    entity_score = entity_overlap(
        entities_a,
        entities_b
    )

    temporal_score = temporal_compatibility(
        date_a,
        date_b
    )

    location_score = location_compatibility(
        location_a,
        location_b
    )

    return [
        similarity,
        entity_score,
        temporal_score,
        location_score
    ]