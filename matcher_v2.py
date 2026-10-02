"""
Variant B event matcher (V4 approach 1, stage 1b) behind /predict/v2 and
/vocabulary/v2.

The backend sends, per candidate event, the centroid similarity, the
temporal score and the four member similarities (computed in SQL over all
members), plus the texts of up to MEMBER_TEXT_CAP newest members. This module
adds the two TF-IDF wording features and scores with the logistic regression
trained by train_event_matcher_v2.py.

The TF-IDF vocabulary is refitted nightly by the backend (POST /vocabulary/v2)
on recent articles; until the first refit after a start, the vocabulary
shipped in the model file is used ("default").
"""

from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from event_features import new_vectorizers, strip_templates, tfidf_text

MODEL_PATH = Path(__file__).parent / "models" / "event_matcher_v2.joblib"
META_PATH = Path(__file__).parent / "models" / "event_matcher_v2.meta.json"
MEMBER_TEXT_CAP = 20
DEFAULT_VOCABULARY = "default"


# ============================================================
# VOCABULARY
# ============================================================

def fit_vocabulary(items):
    """Fit (title, text) vectorisers on [{title, description}] the same way as offline."""
    stripped = [strip_templates(item["title"]) for item in items]
    titles, texts = new_vectorizers()
    titles.fit(stripped)
    texts.fit([tfidf_text(s, item.get("description")) for s, item in zip(stripped, items)])
    return titles, texts


class Matcher:

    def __init__(self, artefact):
        self.model = artefact["model"]
        self.columns = artefact["columns"]
        self.threshold = artefact["threshold"]
        self.model_version = artefact["version"]
        self.default_vocabulary = (artefact["title_vectorizer"], artefact["text_vectorizer"], DEFAULT_VOCABULARY)
        self.vocabulary = self.default_vocabulary

    def refit(self, items):
        """Fit a new vocabulary and swap it in as one reference, so a concurrent
        prediction uses either the old or the new vocabulary, never a mix."""
        titles, texts = fit_vocabulary(items)
        fitted_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.vocabulary = (titles, texts, f"{fitted_at}/{len(items)}")
        return self.vocabulary[2]

    # ============================================================
    # FEATURES AND SCORING
    # ============================================================

    def features(self, title, description, candidates, vocabulary):
        """One feature row per candidate, in the training column names."""
        title_vectorizer, text_vectorizer, _ = vocabulary
        article_title = strip_templates(title)
        member_titles, member_texts, spans = [], [], []
        # Members arrive newest first; only the newest MEMBER_TEXT_CAP texts are used.
        for candidate in candidates:
            members = candidate["members"][:MEMBER_TEXT_CAP]
            start = len(member_titles)
            for member in members:
                stripped = strip_templates(member["title"])
                member_titles.append(stripped)
                member_texts.append(tfidf_text(stripped, member.get("description")))
            spans.append((start, len(member_titles)))

        # TfidfVectorizer rows are L2-normalised, so a dot product is the cosine.
        titles = title_vectorizer.transform([article_title] + member_titles)
        texts = text_vectorizer.transform([tfidf_text(article_title, description)] + member_texts)
        title_sims = (titles[1:] @ titles[0].T).toarray().ravel()
        text_sims = (texts[1:] @ texts[0].T).toarray().ravel()

        rows = []
        for candidate, (start, end) in zip(candidates, spans):
            rows.append({
                "similarity": candidate["similarity"],
                "temporal_score": candidate["temporal_score"],
                "member_max": candidate["member_max"],
                "member_min": candidate["member_min"],
                "member_top3": candidate["member_top3"],
                "member_newest": candidate["member_newest"],
                "title_tfidf_max": float(title_sims[start:end].max()),
                "text_tfidf_max": float(text_sims[start:end].max()),
            })
        return rows

    def predict(self, title, description, candidates):

        vocabulary = self.vocabulary
        results = []
        if candidates:
            rows = self.features(title, description, candidates, vocabulary)
            probabilities = self.model.predict_proba(pd.DataFrame(rows)[self.columns])[:, 1]
            results = [
                {"event_id": candidate["event_id"], "probability": float(probability), "features": row}
                for candidate, probability, row in zip(candidates, probabilities, rows)
            ]
        return {
            "model_version": self.model_version,
            "vocabulary_version": vocabulary[2],
            "threshold": self.threshold,
            "results": results,
        }


def load_matcher(path=MODEL_PATH):
    return Matcher(joblib.load(path))
