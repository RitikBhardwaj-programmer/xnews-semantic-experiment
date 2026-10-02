"""
Numeric cricket claims for the information layer (V4 roadmap step 6),
behind POST /claims. Rules on purpose (no cost, no new dependency) and
precision first: a claim needs a named subject right next to a cricket
number pattern, and the article must look like cricket.

Each claim keeps the exact quote and its position in the title or
description, so a reviewer sees what it rests on. Claims are shown only after
human review (backend).

Predicates and values:
    innings_score     "India ... 351/9"           value 351 runs, value_text "351/9"
    all_out_for       "Pakistan all out for 180"  value 180 runs
    chased            "India chase 406"           value 406 runs
    set_target        "England set a target of 287"
    runs_scored       "Kohli scores 112", "Raichandani's 240"
    bowling_figures   "Chauhan took 6/75"         value 6 wickets, value_text "6/75"
    won_by / lost_by  "beat South Africa by 5 wickets", "lost by 32 runs"
"""

import re

from entity_extraction import NOT_NAMES, normalize
from event_features import _TEAM_RE, STOPWORDS, teams

CRICKET_RE = re.compile(
    r"\b(cricket|odis?|t20is?|t20|test match|wickets?|innings|runs|ipl|bcci|ranji|century|centuries|ton"
    r"|fifty|fifties|half-century|bowled|batting|bowling|pacer|spinner|all-rounder)\b",
    re.IGNORECASE,
)
# "Test" with a capital T is the cricket format; lower-case "test" is not.
CRICKET_TEST_RE = re.compile(r"\bTests?\b")

# A capitalised name of 1-4 words ("India", "Shubman Gill", "West Indies").
NAME = r"(?P<subject>[A-Z][\w'’.&-]*(?:\s+[A-Z][\w'’.&-]*){0,3})"
NOT_UNITS = r"(?!\s*(?:balls|deliveries|overs|years|days|per|%|crore|lakh|km|million|billion|seats|votes))"

PATTERNS = [
    ("bowling_figures", re.compile(
        NAME + r"\s+(?:took|takes|claimed|claims|picked up|picks up|grabbed|grabs|bagged|bags|returned|finished with)\s+"
        r"(?P<a>\d{1,2})/(?P<b>\d{1,3})\b")),
    ("all_out_for", re.compile(
        NAME + r"\s+(?:were\s+|was\s+|are\s+|is\s+|got\s+|get\s+)?(?:all out|bowled out)\s+for\s+(?P<a>\d{2,3})\b")),
    ("chased", re.compile(
        NAME + r"\s+(?:chased|chase|chases|chasing)\s+(?:down\s+)?(?P<a>\d{2,3})\b" + NOT_UNITS)),
    ("set_target", re.compile(
        NAME + r"\s+(?:set|sets|setting)\s+(?:a\s+|an\s+)?(?:\w+\s+)?target of\s+(?P<a>\d{2,3})\b")),
    ("runs_scored", re.compile(
        NAME + r"\s+(?:scored|scores|score|smashed|smashes|hit|hits|struck|made|makes)\s+(?:a\s+|an\s+)?"
        r"(?:unbeaten\s+|brilliant\s+|sublime\s+|career-best\s+|maiden\s+|crucial\s+|quickfire\s+)?(?P<a>\d{2,3})\b"
        + NOT_UNITS)),
    ("runs_scored", re.compile(NAME + r"['’]s\s+(?:unbeaten\s+)?(?P<a>\d{2,3})\b" + NOT_UNITS)),
    ("margin", re.compile(
        NAME + r"\s+(?P<verb>won|win|wins|beat|beats|defeated|defeat|thrashed|crushed|lost|lose|loses)\s+"
        r"(?:[\w'’-]+\s+){0,4}?by\s+(?P<a>\d{1,3})\s+(?P<unit>runs|wickets)\b")),
    ("innings_score", re.compile(
        NAME + r"\s+(?:(?:to|on|at|reach|reaches|reached|post|posts|posted|were|was|finish on|finished on|close on|closed on)\s+)?"
        r"(?P<a>\d{2,3})/(?P<b>\d{1,2})\b")),
]

LOSING = {"lost", "lose", "loses"}


ROLES = {"opener", "openers", "captain", "skipper", "pacer", "spinner", "batter", "batsman", "keeper",
         "wicketkeeper", "all-rounder", "debutant", "veteran", "star"}


def _subject(raw):
    """Clean a subject: drop leading stopwords and roles ('The Aussies' -> 'Aussies',
    'Opener Sam Konstas' -> 'Sam Konstas'); cut to the team when it starts with one
    ('West Indies Notch Record' -> 'West Indies'); reject non-names."""
    words = raw.split()
    while words and words[0].lower().strip(".'’") in STOPWORDS | NOT_NAMES | ROLES:
        words = words[1:]
    if not words or any(w.lower().strip(".'’") in NOT_NAMES for w in words):
        return None
    subject = " ".join(words)
    team = _TEAM_RE.match(subject)
    if team and team.group(1) and (len(team.group(1)) > 3 or team.group(1).isupper()):
        subject = subject[:team.end()].strip()
    return subject


def _is_team(subject):
    return bool(teams(subject)) and normalize(subject) in {normalize(t) for t in teams(subject)} | {
        normalize(m.group(0)) for m in _TEAM_RE.finditer(subject)}


def _claim(predicate, match):
    a = int(match.group("a"))
    if predicate == "bowling_figures":
        wickets, runs = a, int(match.group("b"))
        if wickets > 10:
            return None
        return predicate, wickets, f"{wickets}/{runs}", "wickets"
    if predicate == "innings_score":
        runs, wickets = a, int(match.group("b"))
        if wickets > 10 or runs < 10:
            return None
        return predicate, runs, f"{runs}/{wickets}", "runs"
    if predicate == "margin":
        unit = match.group("unit").lower()
        name = "lost_by" if match.group("verb").lower() in LOSING else "won_by"
        if unit == "wickets" and a > 10:
            return None
        return name, a, f"{a} {unit}", unit
    return predicate, a, str(a), "runs"


def claims(title, description):
    """[{subject, subject_normalized, predicate, value, value_text, unit, quote, start, end, field}]"""
    texts = (("title", title or ""), ("description", description or ""))
    joined = " ".join(t for _, t in texts)
    if not (CRICKET_RE.search(joined) or CRICKET_TEST_RE.search(joined)):
        return []

    result, seen = [], set()
    for field, text in texts:
        for kind, pattern in PATTERNS:
            for match in pattern.finditer(text):
                subject = _subject(match.group("subject"))
                parsed = _claim(kind, match)
                if subject is None or parsed is None:
                    continue
                predicate, value, value_text, unit = parsed
                # "Sri Lanka's 45" is a team total, not a player's runs.
                if predicate == "runs_scored" and _is_team(subject):
                    continue
                key = (normalize(subject), predicate, value_text, field)
                if key in seen:
                    continue
                seen.add(key)
                # The quote starts at the subject as kept (leading "The" dropped).
                start = match.start("subject") + match.group("subject").index(subject.split()[0])
                result.append({
                    "subject": subject,
                    "subject_normalized": normalize(subject),
                    "predicate": predicate,
                    "value": value,
                    "value_text": value_text,
                    "unit": unit,
                    "quote": text[start:match.end()],
                    "start": start,
                    "end": match.end(),
                    "field": field,
                })
    return result
