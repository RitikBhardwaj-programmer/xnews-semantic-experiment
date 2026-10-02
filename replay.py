"""
Free-running replay of production event matching (V4 approach 1, stage 0).

Mirrors ArticleProcessingService.matchOrCreateEvent in the backend, but
keeps its own mistakes (unlike the oracle replay in
evaluate_centroid_matching.py, which resets to the true event after every
decision):

- articles in processing order (created_at, id), state starts empty
- candidates: OPEN events (last activity within INACTIVITY_DAYS of "now",
  the article's created_at), top CANDIDATE_LIMIT by centroid cosine
- features [similarity to centroid, temporal_score of the day gap between the
  article's published date and the event's last activity]; missing dates -> 0.5
- the best-probability candidate attaches if probability >= threshold,
  otherwise a new event is created
- running-mean centroid; first/last activity from published_at
  (NewsEvent.addArticle)

The attach decision is a function (`decide`), so approach-1 variants can be
compared on the same harness.
"""

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from evaluate_centroid_matching import FEATURES, MODEL_PATH, temporal_score, update_centroid

CANDIDATE_LIMIT = 30        # ai.event-matcher.candidate-limit
INACTIVITY_DAYS = 10        # xnews.event.inactivity-days
THRESHOLD = 0.94            # ai.event-matcher.threshold


def load_day(path):

    articles = pd.read_csv(path, parse_dates=["published_at", "created_at"])
    vectors = np.stack(articles["embedding"].map(
        lambda s: np.array([float(x) for x in s.strip("[]").split(",")], dtype=np.float64)
    ).values)
    articles["vector"] = list(vectors / np.linalg.norm(vectors, axis=1, keepdims=True))

    return articles.drop(columns=["embedding"])


@dataclass
class Event:
    centroid: np.ndarray
    members: list = field(default_factory=list)
    first_activity: pd.Timestamp = None
    last_activity: pd.Timestamp = None


def day_gap(published, last_activity):
    if pd.isna(published) or pd.isna(last_activity):
        return None
    return abs((published.normalize() - last_activity.normalize()).days)


def model_decide(model, threshold):
    """Production rule: best probability among candidates, attach if >= threshold."""

    def decide(article, candidates):
        if not candidates:
            return None
        frame = pd.DataFrame(
            [(c["similarity"], c["temporal_score"]) for c in candidates], columns=FEATURES
        )
        probability = model.predict_proba(frame)[:, 1]
        best = int(np.argmax(probability))
        return candidates[best]["index"] if probability[best] >= threshold else None

    return decide


def replay(articles, decide):
    """Returns the predicted event index of every article (same order as `articles`)."""

    order = articles.sort_values(["created_at", "id"]).index
    events = []
    assigned = pd.Series(-1, index=articles.index)

    for row in order:

        article = articles.loc[row]
        vector = article["vector"]
        now = article["created_at"]

        open_events = [
            k for k, e in enumerate(events)
            if e.last_activity >= now - pd.Timedelta(days=INACTIVITY_DAYS)
        ]

        candidates = []
        if open_events:
            sims = np.stack([events[k].centroid for k in open_events]) @ vector
            for j in np.argsort(-sims)[:CANDIDATE_LIMIT]:
                k = open_events[j]
                gap = day_gap(article["published_at"], events[k].last_activity)
                candidates.append({
                    "index": k,
                    "similarity": float(sims[j]),
                    "temporal_score": 0.5 if gap is None else float(temporal_score(np.array(gap))),
                })

        chosen = decide(article, candidates)

        if chosen is None:
            event = Event(centroid=update_centroid(None, 0, vector),
                          first_activity=now, last_activity=now)
            events.append(event)
            chosen = len(events) - 1
        else:
            event = events[chosen]
            event.centroid = update_centroid(event.centroid, len(event.members), vector)

        event.members.append(row)
        published = article["published_at"]
        if pd.notna(published):
            if len(event.members) == 1:
                event.first_activity = event.last_activity = published
            else:
                event.first_activity = min(event.first_activity, published)
                event.last_activity = max(event.last_activity, published)

        assigned[row] = chosen

    return assigned.to_numpy()


def production_replay(articles, threshold=THRESHOLD):
    return replay(articles, model_decide(joblib.load(MODEL_PATH), threshold))

