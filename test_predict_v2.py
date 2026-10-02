"""
Checks the v2 matcher behind /predict/v2 and /vocabulary/v2.

- Parity: for oracle-replay candidates from a labelled day, the service code
  reproduces the offline features (event_features.ArticleIndex) and model
  probabilities to 1e-9. Needs the local descriptions; skipped without them.
- Vocabulary refit, empty requests and the member text cap.

Run: python test_predict_v2.py
"""

import numpy as np
import pandas as pd

from evaluate_stage1 import BASE, DAYS, load
from event_features import ArticleIndex, strip_templates, tfidf_text
from matcher_v2 import DEFAULT_VOCABULARY, MEMBER_TEXT_CAP, load_matcher
from replay import replay

MEMBER_COLUMNS = ["member_max", "member_min", "member_top3", "member_newest"]


def synthetic_items(n, word="volcano"):
    return [{"title": f"{word} erupts near village number {i}", "description": f"ash cloud report {i}"}
            for i in range(n)]


def candidate(event_id, members, similarity=0.9):
    return {"event_id": event_id, "similarity": similarity, "temporal_score": 1.0,
            "member_max": 0.9, "member_min": 0.8, "member_top3": 0.85, "member_newest": 0.9,
            "members": members}


def test_parity_with_offline_features():
    if not (BASE / "local" / "descriptions.csv").exists():
        print("skipped parity: local descriptions missing")
        return
    matcher = load_matcher()
    titles, texts, _ = matcher.default_vocabulary
    articles = load(DAYS[1])
    index = ArticleIndex(articles.assign(title=articles["stripped"]), titles.transform(articles["stripped"]),
                         texts.transform([tfidf_text(s, d if d == d else None)
                                          for s, d in zip(articles["stripped"], articles["description"])]),
                         MEMBER_TEXT_CAP)

    # Oracle replay: follow the true event, record candidates (large events included, to cover the cap).
    samples, event_of_truth, rng = [], {}, np.random.default_rng(0)

    def decide(article, candidates):
        for c in candidates:
            if len(c["members"]) > MEMBER_TEXT_CAP or rng.random() < 0.002:
                samples.append((article.name, dict(c, members=list(c["members"]))))
        true_event = event_of_truth.get(article["truth"])
        if true_event is None:
            event_of_truth[article["truth"]] = decide.created
            decide.created += 1
        return true_event

    decide.created = 0
    replay(articles, decide)
    large = [s for s in samples if len(s[1]["members"]) > MEMBER_TEXT_CAP]
    small = [s for s in samples if len(s[1]["members"]) <= MEMBER_TEXT_CAP]
    chosen = [large[i] for i in rng.choice(len(large), 25, replace=False)] + \
             [small[i] for i in rng.choice(len(small), 25, replace=False)]

    for row, c in chosen:
        expected = index.features(row, c["members"], c["similarity"], c["temporal_score"])
        expected = {k: expected[k] for k in matcher.columns}
        expected_p = matcher.model.predict_proba(pd.DataFrame([expected])[matcher.columns])[:, 1][0]

        def text(m):
            d = articles.loc[m, "description"]
            return {"title": articles.loc[m, "title"], "description": d if d == d else None}

        request = candidate(7, [text(m) for m in reversed(c["members"])][:MEMBER_TEXT_CAP], c["similarity"])
        request.update(temporal_score=c["temporal_score"], **{k: expected[k] for k in MEMBER_COLUMNS})
        description = articles.loc[row, "description"]
        got = matcher.predict(articles.loc[row, "title"], description if description == description else None,
                              [request])
        result = got["results"][0]
        for k in matcher.columns:
            assert abs(result["features"][k] - expected[k]) < 1e-9, (row, k, result["features"][k], expected[k])
        assert abs(result["probability"] - expected_p) < 1e-9
    print(f"parity: {len(chosen)} candidates ({len(large)} large-event samples available)")


def test_empty_candidates():
    result = load_matcher().predict("Volcano erupts in Iceland", None, [])
    assert result["results"] == [] and result["vocabulary_version"] == DEFAULT_VOCABULARY
    assert 0 < result["threshold"] < 1 and result["model_version"].startswith("v2")


def test_refit_swaps_vocabulary():
    matcher = load_matcher()
    members = [{"title": "Zyxqar eruption near the village forces evacuation", "description": None}]
    before = matcher.predict("Zyxqar eruption forces evacuation of village", None, [candidate(1, members)])
    version = matcher.refit(synthetic_items(150, word="zyxqar"))
    after = matcher.predict("Zyxqar eruption forces evacuation of village", None, [candidate(1, members)])
    assert version != DEFAULT_VOCABULARY and version.endswith("/150")
    assert after["vocabulary_version"] == version != before["vocabulary_version"]
    # "zyxqar" is unknown to the default vocabulary, known after the refit
    assert after["results"][0]["features"]["title_tfidf_max"] != before["results"][0]["features"]["title_tfidf_max"]


def test_refit_rejects_stop_words_only():
    matcher = load_matcher()
    try:
        matcher.refit([{"title": "the and of", "description": "it is"}] * 150)
    except ValueError:
        assert matcher.vocabulary is matcher.default_vocabulary
        return
    raise AssertionError("a stop-word-only vocabulary should be rejected")


def test_member_text_cap_uses_first_members():
    """Members arrive newest first; only the first MEMBER_TEXT_CAP feed the TF-IDF features."""
    matcher = load_matcher()
    match = {"title": "Kohli scores century as India beat West Indies in second ODI", "description": None}
    other = {"title": "Monsoon floods close schools across Kerala districts", "description": None}
    title = "Kohli century as India beat West Indies in second ODI"
    capped = matcher.predict(title, None, [candidate(1, [other] * MEMBER_TEXT_CAP + [match])])
    included = matcher.predict(title, None, [candidate(1, [match] + [other] * MEMBER_TEXT_CAP)])
    assert capped["results"][0]["features"]["title_tfidf_max"] < 0.2
    assert included["results"][0]["features"]["title_tfidf_max"] > 0.5
    assert strip_templates(title) == title


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} tests passed")
