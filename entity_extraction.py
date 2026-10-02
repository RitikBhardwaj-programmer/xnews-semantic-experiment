"""
Entity mentions for the information layer (V4 roadmap step 3), behind
POST /entities. Rule-based on purpose (no new dependency): teams from the
gazetteer in event_features, plus runs of capitalised words as candidate
names of people, places and organisations.

Stricter than event_features.entities(), which serves as a matcher feature:
- a name never runs across punctuation ("Kohli, Gill" is two names)
- a title-case headline ("Heavy Rain Lashes Chennai") carries no capitalisation
  signal, so its title is skipped and only the description is used
- format words (ODI, T20I, LIVE), weekdays and months are not names
- a name is at most 5 words ("Chief Election Commissioner Gyanesh Kumar")
- a team inside a longer name ("India Meteorological Department") is not a team
"""

import re

from event_features import STOPWORDS, teams

MAX_NAME_WORDS = 5
TITLE_CASE_SHARE = 0.7

NOT_NAMES = set("""
odi odis t20 t20i t20is test tests live updates update breaking watch video photos explained opinion
monday tuesday wednesday thursday friday saturday sunday
january february march april may june july august september october november december
day week year today yesterday tomorrow
""".split())

_SEGMENT_RE = re.compile(r"[,;:|!?()\[\]\"“”‘’–—]|\s-\s|\.\s")
_WORD_RE = re.compile(r"[A-Za-z][\w'.&-]*")


def normalize(name):
    """Alias key: lower case, letters/digits/& only, single spaces."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w& ]", " ", name.lower())).strip()


def is_title_case(text):
    words = [w for w in _WORD_RE.findall(text) if len(w) > 3]
    return bool(words) and sum(w[0].isupper() for w in words) / len(words) >= TITLE_CASE_SHARE


def _names(text):
    found = []
    for segment in _SEGMENT_RE.split(text):
        words = _WORD_RE.findall(segment)
        run = []
        for position, word in enumerate(words + [""]):
            word = re.sub(r"'s$", "", word)  # possessive: "India's" -> "India"
            clean = word.strip(".'").lower()
            if word[:1].isupper() and clean not in STOPWORDS and clean not in NOT_NAMES:
                run.append(word.strip(".'"))
                continue
            # A capitalised first word of a segment only counts as part of a run.
            if len(run) > 1 or (run and position - len(run) > 0):
                if len(run) <= MAX_NAME_WORDS:
                    found.append(" ".join(run))
            run = []
    return found


def mentions(title, description):
    """[{text, normalized, type, field}], one per distinct entity and field."""
    result, seen = [], set()

    def add(text, kind, field):
        key = normalize(text)
        if len(key) > 2 and (key, field) not in seen:
            seen.add((key, field))
            result.append({"text": text, "normalized": key, "type": kind, "field": field})

    for field, text in (("title", title or ""), ("description", description or "")):
        names = [] if field == "title" and is_title_case(text) else _names(text)
        name_keys = [normalize(n) for n in names]
        team_keys = set()
        for team in teams(text):
            key = normalize(team)
            team_keys.add(key)
            # "India" at the start of "India Meteorological Department" is not the team.
            if not any(n != key and n.startswith(key + " ") for n in name_keys):
                add(team.title(), "team", field)
        for name, key in zip(names, name_keys):
            if key not in team_keys:
                add(name, "name", field)

    return result
