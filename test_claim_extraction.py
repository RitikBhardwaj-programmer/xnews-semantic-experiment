"""
Checks claim_extraction (POST /claims) on real cricket sentence shapes.
Run: python test_claim_extraction.py
"""

from claim_extraction import claims


def found(title, description=None):
    return {(c["subject"], c["predicate"], c["value_text"]) for c in claims(title, description)}


def test_bowling_figures_and_player_runs():
    result = found("Lakshya Raichandani's Double Ton Sets Up Dominant India U-19 Victory Against Australia",
                   "Yashbardhan Chauhan took 6/75 while Ishan Om claimed 4/62 after Lakshya Raichandani's 240 "
                   "as India U-19 crushed their Australian counterparts.")
    assert {("Yashbardhan Chauhan", "bowling_figures", "6/75"), ("Ishan Om", "bowling_figures", "4/62"),
            ("Lakshya Raichandani", "runs_scored", "240")} <= result


def test_innings_score_and_margin():
    assert ("India", "innings_score", "351/9") in found(
        "IND-A vs AUS A Unofficial Test Day 1: Tanisha Singh 92, Nikki, Shubha fifties guide India to 351/9")
    assert ("Aussies", "lost_by", "32 runs") in found(
        "Big Blow For Australia", "Star pacer suffered a side strain in the second ODI, which the Aussies lost by 32 runs.")
    assert ("India", "won_by", "5 wickets") in found("India beat West Indies by 5 wickets in the second ODI")


def test_chase_target_and_all_out():
    assert ("India", "chased", "406") in found("Gill on the double as India chase 406 in style in the ODI")
    assert ("England", "set_target", "287") in found("England set a stiff target of 287 in the first ODI")
    assert ("Pakistan", "all_out_for", "180") in found("Pakistan all out for 180 on day two of the Test match")


def test_only_cricket_and_plausible_numbers():
    assert claims("Prahaar trailer: Rajkummar Rao plays the lawyer in the 26/11 Mumbai attacks case", None) == []
    assert found("Kohli scores 112 off 90 balls in the ODI") == {("Kohli", "runs_scored", "112")}
    assert found("India 26/11 in the T20 match") == set()          # 11 wickets is not a score
    assert ("India", "runs_scored", "45") not in found("India hit 45 crore in ODI sponsorship")
    # Found in a local run: a ball count read as runs.
    assert found("Shubman Gill's 117-ball double century tops the list of fastest ODI 200s") == set()


def test_subjects_are_cleaned():
    assert ("West Indies", "innings_score", "405/7") in found("Cricket: West Indies Notch Record 405/7 In The ODI")
    assert ("Sam Konstas", "runs_scored", "40") in found("x", "Opener Sam Konstas hit 40 before lunch on day one of the Test.")
    # A team's number after "'s" is a total, not a player's runs.
    assert found("x", "Sri Lanka's 45 was their lowest ODI total.") == set()


def test_quote_points_at_the_text():
    title = "Big Blow For Australia"
    description = "Star pacer suffered a side strain in the second ODI, which the Aussies lost by 32 runs."
    claim = next(c for c in claims(title, description) if c["predicate"] == "lost_by")
    assert claim["field"] == "description"
    assert description[claim["start"]:claim["end"]] == claim["quote"] == "Aussies lost by 32 runs"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} tests passed")
