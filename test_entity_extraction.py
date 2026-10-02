"""
Checks entity_extraction (POST /entities) on real headline shapes.
Run: python test_entity_extraction.py
"""

from entity_extraction import is_title_case, mentions, normalize


def found(title, description=None, field=None):
    return {(m["text"], m["type"]) for m in mentions(title, description) if field in (None, m["field"])}


def test_teams_and_full_names():
    result = found("IND vs WI, 2nd ODI: Kohli, Gill in action as India looks to seal series",
                   "Virat Kohli and Shubman Gill star as India chase 406 at Ahmedabad.")
    assert {("India", "team"), ("West Indies", "team"), ("Virat Kohli", "name"),
            ("Shubman Gill", "name"), ("Ahmedabad", "name")} <= result
    assert not any("Odi" in text or "ODI" in text for text, _ in result)


def test_names_do_not_run_across_punctuation():
    names = {text for text, _ in found("Supreme Court quashes NSA detention of Sambhal violence accused Mulla Afroz")}
    assert names == {"Supreme Court", "NSA", "Sambhal", "Mulla Afroz"}


def test_long_titles_of_people_are_kept():
    result = found("x", "The Congress leader met Chief Election Commissioner Gyanesh Kumar on Monday.")
    assert ("Chief Election Commissioner Gyanesh Kumar", "name") in result
    assert not any(text == "Monday" for text, _ in result)


def test_title_case_headline_uses_the_description_only():
    assert is_title_case("Heavy Rain Lashes Chennai, Schools Shut On Monday")
    result = found("Heavy Rain Lashes Chennai, Schools Shut On Monday",
                   "The India Meteorological Department issued an orange alert for Chennai and Tiruvallur.")
    assert {text for text, _ in result} == {"India Meteorological Department", "Chennai", "Tiruvallur"}


def test_one_mention_per_entity_and_field():
    result = mentions("Kerala floods: Kerala CM visits Wayanad", "Kerala CM visits Wayanad again.")
    keys = [(m["normalized"], m["field"]) for m in result]
    assert len(keys) == len(set(keys))


def test_possessive_is_not_part_of_the_name():
    assert ("Rohit Sharma", "name") in found("x", "Captain Gill praised Rohit Sharma's knock at Eden Gardens.")


def test_normalize():
    assert normalize("  J&K's  Omar-Abdullah ") == "j&k s omar abdullah"
    assert mentions(None, None) == []


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} tests passed")
