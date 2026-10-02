"""
Event-evidence features for the article -> candidate event decision
(V4 approach 1, stage 1). Pure functions; standard library, numpy and
scikit-learn only.

The original matcher sees one number: cosine to a running-mean centroid. These
features add evidence that separates "same event" from "same topic": closeness
to actual members, title wording, people/teams/places, numbers, the cricket
fixture a headline is about, and time.
"""

import math
import re
from datetime import timedelta

import numpy as np

# ------------------------------------------------------------
# TEMPLATE STRIPPING
# ------------------------------------------------------------

TEMPLATES = [
    r"\blive\s+(?:score|scores|updates?|streaming(?:\s+info)?|telecast|blog|coverage)\b",
    r"\bfree\s+live\s+telecast\b",
    r"\b(?:when|and|,|\s)*where\s+to\s+watch\b(?:\s+\w+){0,6}",
    r"\bcheck\s+(?:details|here|full\s+list|important\s+dates|direct\s+link)\b(?:\s+here)?",
    r"\bfull\s+list\b",
    r"\|\s*explained\b",
    r"\bexplained\b",
    r"\blive\b\s*:?",
    r"\s\|\s.*$",
]
_TEMPLATE_RE = re.compile("|".join(TEMPLATES), re.IGNORECASE)


def strip_templates(title):
    """Remove recurring headline boilerplate; keep the original if too little is left."""
    stripped = _TEMPLATE_RE.sub(" ", title)
    stripped = re.sub(r"[\s:;,–—-]+$", "", re.sub(r"\s+", " ", stripped)).strip(" :;,–—-")
    return stripped if len(stripped.split()) >= 3 else title


# ------------------------------------------------------------
# ENTITIES
# ------------------------------------------------------------

# Team names normalised to one form (cricket and other national sides that
# appear in the feeds). Longer names first so "West Indies" wins over "Indies".
TEAMS = {
    "india": "india", "ind": "india",
    "west indies": "west indies", "windies": "west indies", "wi": "west indies",
    "sri lanka": "sri lanka", "sl": "sri lanka",
    "pakistan": "pakistan", "pak": "pakistan",
    "australia": "australia", "aus": "australia",
    "south africa": "south africa", "sa": "south africa",
    "england": "england", "eng": "england",
    "new zealand": "new zealand", "nz": "new zealand",
    "bangladesh": "bangladesh", "ban": "bangladesh",
    "afghanistan": "afghanistan", "afg": "afghanistan",
    "zimbabwe": "zimbabwe", "zim": "zimbabwe",
    "ireland": "ireland", "ire": "ireland",
    "japan": "japan", "nepal": "nepal", "qatar": "qatar", "brazil": "brazil",
    "jammu and kashmir": "jammu and kashmir", "j&k": "jammu and kashmir",
    "rest of india": "rest of india",
}
_TEAM_RE = re.compile(
    r"(?<![\w&])(" + "|".join(sorted((re.escape(k) for k in TEAMS), key=len, reverse=True))
    + r")(?:\s*-\s*|\s+)?(a\b|u-?19\b|under-?19\b|women\b|women's\b)?(?![\w&])",
    re.IGNORECASE,
)

STOPWORDS = set("""
a an the and or but of in on at to for from by with as is are was were be been it its this that
these those he she they we you i his her their our your who what when where why how not no yes
after before over under into about than then so if up down out off new more most all any some
says said say amid vs v live watch score scores updates update check details full list day
""".split())


def teams(text):
    """Normalised teams mentioned, with side markers ('india a', 'india u19', 'india women')."""
    found = []
    for match in _TEAM_RE.finditer(text):
        name, side = match.group(1), match.group(2)
        team = TEAMS[name.lower()]
        # Short abbreviations only count when written in capitals (avoids "sa", "ban" as words).
        if len(name) <= 3 and not name.isupper() and name.lower() not in ("j&k",):
            continue
        if side:
            side = side.lower().replace("under-", "u").replace("under", "u").replace("-", "").replace("'s", "")
            team = f"{team} {side}"
        found.append(team)
    return found


def entities(text):
    """Teams plus capitalised name runs (people, places, organisations), lower-cased."""
    found = set(teams(text))
    words = re.findall(r"[A-Za-z][\w'’.-]*", text)
    run = []
    for position, word in enumerate(words + [""]):
        clean = word.strip(".'’").lower()
        capitalised = word[:1].isupper() and clean not in STOPWORDS
        # The first word of a headline is capitalised anyway, so it only counts
        # when it is part of a longer run.
        if capitalised:
            run.append(clean)
            continue
        if len(run) > 1 or (run and position - len(run) > 0):
            found.add(" ".join(run))
        run = []
    return {e for e in found if len(e) > 2}


# ------------------------------------------------------------
# NUMBERS AND FIXTURES
# ------------------------------------------------------------

ORDINALS = {
    "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4, "fifth": 5, "5th": 5,
}
_FORMAT_RE = re.compile(
    r"\b(odi|one-day international|t20i|t20|test|semi-?finals?|final|quarter-?finals?)\b", re.IGNORECASE
)
_ORDINAL_FORMAT_RE = re.compile(
    r"\b(first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th)\b(?:\s+\w+){0,4}?\s+"
    r"(odi|one-day international|t20i|t20|test)\b",
    re.IGNORECASE,
)


def numbers(text):
    """Numbers that identify an event: scores, amounts, match numbers (not years)."""
    found = set(re.findall(r"\b\d+(?:[.,]\d+)*(?:/\d+)?\b", text))
    found |= {f"#{ORDINALS[m.group(1).lower()]}" for m in _ORDINAL_FORMAT_RE.finditer(text)}
    return {n for n in found if not re.fullmatch(r"(19|20)\d\d", n)}


def _format(raw):
    raw = raw.lower().replace("-", "")
    if raw.startswith("semi"):
        return "semifinal"
    if raw.startswith("quarter"):
        return "quarterfinal"
    return {"one-day international": "odi", "onedayinternational": "odi", "t20": "t20i"}.get(raw, raw)


def match_signature(text):
    """(teams, format, number) for a headline about a fixture, else None.

    A fixture needs two distinct teams, or one team plus a numbered match
    ("3rd ODI vs West Indies"). Format and number are None when the headline
    doesn't state them.
    """
    sides = sorted(set(teams(text)))
    ordinal = _ORDINAL_FORMAT_RE.search(text)
    fmt = _FORMAT_RE.search(text)
    if len(sides) < 2 and not (sides and ordinal):
        return None
    return (
        tuple(sides[:2]),
        _format(ordinal.group(2) if ordinal else fmt.group(1)) if (ordinal or fmt) else None,
        ORDINALS[ordinal.group(1).lower()] if ordinal else None,
    )


def signature_conflict(a, b):
    """True when two headlines are clearly about different fixtures."""
    if a is None or b is None:
        return False
    # One side may name only one team ("3rd ODI vs West Indies"): compatible
    # as long as one team set contains the other.
    if not (set(a[0]) <= set(b[0]) or set(b[0]) <= set(a[0])):
        return True
    if a[1] and b[1] and a[1] != b[1]:
        return True
    return bool(a[2] and b[2] and a[2] != b[2])


# ------------------------------------------------------------
# ARTICLE -> CANDIDATE EVENT FEATURES
# ------------------------------------------------------------

FEATURE_GROUPS = {
    "base": ["similarity", "temporal_score"],
    "members": ["member_max", "member_min", "member_top3", "member_newest"],
    "lexical": ["title_tfidf_max", "text_tfidf_max"],
    "entities": ["entity_jaccard_max", "entity_jaccard_union", "number_overlap"],
    "time": ["hours_newest", "hours_mean", "hours_oldest", "log_size", "age_hours"],
}


def _jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def _hours(later, earlier):
    if later is None or earlier is None or later != later or earlier != earlier:
        return 48.0
    return abs((later - earlier).total_seconds()) / 3600


class ArticleIndex:
    """Per-article precomputed signals for one or more days, addressed by row position."""

    def __init__(self, articles, title_tfidf, text_tfidf):
        self.vectors = np.stack(articles["vector"].values)
        self.title_tfidf = title_tfidf
        self.text_tfidf = text_tfidf
        titles = articles["title"].fillna("")
        descriptions = articles["description"].fillna("") if "description" in articles else titles * 0
        self.entities = [entities(t) | entities(d) for t, d in zip(titles, descriptions)]
        self.numbers = [numbers(t) for t in titles]
        self.signatures = [match_signature(t) for t in titles]
        self.seen = [p if p == p else c for p, c in zip(articles["published_at"], articles["created_at"])]

    def conflicts(self, row, members):
        """Different fixture than any member that has a signature."""
        return any(signature_conflict(self.signatures[row], self.signatures[m]) for m in members)

    def features(self, row, members, similarity, temporal_score):
        vector = self.vectors[row]
        sims = self.vectors[members] @ vector
        top = np.sort(sims)[::-1]
        title_sims = (self.title_tfidf[members] @ self.title_tfidf[row].T).toarray().ravel()
        text_sims = (self.text_tfidf[members] @ self.text_tfidf[row].T).toarray().ravel()
        union = set().union(*(self.entities[m] for m in members))
        member_numbers = set().union(*(self.numbers[m] for m in members))
        seen = [self.seen[m] for m in members]
        now = self.seen[row]
        valid = [s for s in seen if s == s and s is not None]
        newest = max(valid) if valid else None
        oldest = min(valid) if valid else None
        mean = oldest + sum((s - oldest for s in valid), timedelta(0)) / len(valid) if valid else None
        return {
            "similarity": similarity,
            "temporal_score": temporal_score,
            "member_max": float(top[0]),
            "member_min": float(top[-1]),
            "member_top3": float(top[:3].mean()),
            "member_newest": float(sims[-1]),
            "title_tfidf_max": float(title_sims.max()),
            "text_tfidf_max": float(text_sims.max()),
            "entity_jaccard_max": max(_jaccard(self.entities[row], self.entities[m]) for m in members),
            "entity_jaccard_union": _jaccard(self.entities[row], union),
            "number_overlap": float(bool(self.numbers[row] & member_numbers)),
            "hours_newest": _hours(now, newest),
            "hours_mean": _hours(now, mean),
            "hours_oldest": _hours(now, oldest),
            "log_size": math.log(len(members)),
            "age_hours": _hours(newest, oldest),
        }
