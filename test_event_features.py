"""
Checks event_features on real headline shapes from the labelled days.
Run: python test_event_features.py
"""

from event_features import entities, match_signature, numbers, signature_conflict, strip_templates, teams


def test_strip_live_score_template():
    title = "IND vs WI, 2nd ODI LIVE score: Kohli, Gill, Rohit in action as India looks to seal series"
    stripped = strip_templates(title)
    assert "live" not in stripped.lower()
    assert stripped.startswith("IND vs WI, 2nd ODI") and stripped.endswith("seal series")


def test_strip_where_to_watch():
    title = "IND vs WI, 2nd ODI LIVE streaming info: When, where to watch India vs West Indies match?"
    stripped = strip_templates(title)
    assert "watch" not in stripped.lower() and "streaming" not in stripped.lower()
    assert stripped.startswith("IND vs WI, 2nd ODI")


def test_strip_keeps_title_when_too_little_left():
    assert strip_templates("Live updates") == "Live updates"


def test_teams_and_sides():
    assert teams("IND vs WI, 2nd ODI") == ["india", "west indies"]
    assert teams("IND-A vs AUS A Unofficial Test Day 1") == ["india a", "australia a"]
    assert teams("IND vs AUS, U-19 Test Day 3") == ["india", "australia"]
    assert "west indies" in teams("Windies needs a win")
    assert teams("sa ban on plastic") == []          # lower-case abbreviations aren't teams


def test_signature_same_match_different_wording():
    a = match_signature("IND vs WI, 2nd ODI LIVE score: Kohli in action")
    b = match_signature("India vs West Indies second ODI: Windies needs a win to keep series alive")
    assert a == (("india", "west indies"), "odi", 2)
    assert a == b and not signature_conflict(a, b)


def test_signature_different_match_number():
    second = match_signature("IND vs WI, 2nd ODI: India opt to bowl")
    third = match_signature("Prasidh ruled out of 3rd ODI vs West Indies")
    assert third == (("west indies",), "odi", 3)       # one team plus a numbered match
    assert signature_conflict(second, third)
    assert not signature_conflict(third, match_signature("India vs West Indies third ODI preview"))


def test_signature_different_teams_and_sports():
    cricket = match_signature("IND vs SL weather report: Will rain play spoilsport in Asian Games semifinal?")
    football = match_signature("India Vs Brazil Friendly: Kolkata Gears Up For Big Match")
    assert cricket == (("india", "sri lanka"), "semifinal", None)
    assert signature_conflict(cricket, football)


def test_no_signature_without_two_teams():
    assert match_signature("Virat Kohli builds his ultimate ODI batter") is None
    assert not signature_conflict(None, match_signature("IND vs WI, 2nd ODI"))


def test_missing_number_is_compatible():
    a = match_signature("India vs West Indies: World Cup the target")
    b = match_signature("IND vs WI, 2nd ODI: Auqib Nabi makes debut")
    assert not signature_conflict(a, b)


def test_entities_and_numbers():
    found = entities("Supreme Court quashes NSA detention of Sambhal violence accused Mulla Afroz")
    assert "mulla afroz" in found and "supreme court" in found
    assert numbers("India chase 406, Gill 223 in 2nd ODI of 2026") >= {"406", "223", "#2"}
    assert "2026" not in numbers("Asian Games 2026 results")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} tests passed")
