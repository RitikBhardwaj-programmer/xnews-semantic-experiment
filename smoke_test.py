"""Smoke test for a running AI service.

Checks health, API-key protection, /embed, /predict, /predict/v2 and
/vocabulary/v2 end to end, so a broken model file, dependency or endpoint
fails CI instead of production. It refits the v2 vocabulary, so run it only
against a local or CI container, never the deployed service.

Usage (server already running):
    AI_SERVICE_API_KEY=... python smoke_test.py [base_url]
"""

import json
import math
import os
import sys
import urllib.error
import urllib.request

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
API_KEY = os.environ["AI_SERVICE_API_KEY"]

SAME_STORY = "Jaiswal scores a century as India dominate the second Test"
OTHER_STORY = "Volcano erupts in Iceland, flights diverted across Europe"


def call(method, path, body=None, api_key=API_KEY):
    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["x-api-key"] = api_key
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(BASE_URL + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, None


def check(condition, message):
    if not condition:
        sys.exit(f"FAIL: {message}")
    print(f"ok: {message}")


def embed(text):
    status, body = call("POST", "/embed", {"text": text})
    check(status == 200, f"/embed returns 200 for {text[:30]!r}")
    return body["embedding"]


status, body = call("GET", "/health", api_key=None)
check(status == 200 and body == {"status": "healthy"}, "/health is healthy")

status, _ = call("POST", "/embed", {"text": "x"}, api_key="wrong-key")
check(status == 401, "/embed rejects a wrong API key with 401")

status, _ = call("POST", "/embed", {"text": "x"}, api_key=None)
check(status in (401, 422), "/embed rejects a missing API key")

article = embed(SAME_STORY)
check(len(article) == 384, "embedding has 384 dimensions")
check(abs(math.sqrt(sum(v * v for v in article)) - 1) < 1e-3, "embedding is L2-normalized")

unrelated = embed(OTHER_STORY)

status, body = call("POST", "/predict", {
    "article_embedding": article,
    "candidates": [
        {"event_id": 1, "centroid_embedding": article, "temporal_score": 1.0},
        {"event_id": 2, "centroid_embedding": unrelated, "temporal_score": 1.0},
    ],
})
check(status == 200, "/predict returns 200")

scores = {result["event_id"]: result["probability"] for result in body["results"]}
check(set(scores) == {1, 2}, "/predict scores every candidate")
check(all(0 <= p <= 1 for p in scores.values()), "probabilities are within [0, 1]")
check(scores[1] > scores[2], "the same story scores higher than an unrelated one")

v1_scores = scores


# ------------------------------------------------------------
# /predict/v2 and /vocabulary/v2
# ------------------------------------------------------------

def v2_candidate(event_id, members, similarity):
    return {"event_id": event_id, "similarity": similarity, "temporal_score": 1.0,
            "member_max": similarity, "member_min": similarity, "member_top3": similarity,
            "member_newest": similarity, "members": members}


same_members = [{"title": "Jaiswal hits a hundred, India on top in second Test", "description": None}]
other_members = [{"title": OTHER_STORY, "description": "Air traffic disrupted across Europe."}]
v2_request = {
    "title": SAME_STORY,
    "description": "India batted all day.",
    "candidates": [v2_candidate(1, same_members, 0.95), v2_candidate(2, other_members, 0.1)],
}

status, _ = call("POST", "/predict/v2", v2_request, api_key="wrong-key")
check(status == 401, "/predict/v2 rejects a wrong API key with 401")

status, body = call("POST", "/predict/v2", v2_request)
check(status == 200, "/predict/v2 returns 200")
check(body["vocabulary_version"] == "default", "/predict/v2 starts with the shipped vocabulary")
check(body["model_version"].startswith("v2") and 0 < body["threshold"] < 1, "/predict/v2 reports model and threshold")
v2_scores = {result["event_id"]: result["probability"] for result in body["results"]}
check(set(v2_scores) == {1, 2}, "/predict/v2 scores every candidate")
check(v2_scores[1] > v2_scores[2], "/predict/v2: the same story scores higher than an unrelated one")
check(len(body["results"][0]["features"]) == 8, "/predict/v2 returns the 8 feature values")

status, _ = call("POST", "/predict/v2", dict(v2_request, candidates=[v2_request["candidates"][0]] * 31))
check(status == 422, "/predict/v2 rejects more than 30 candidates")
status, _ = call("POST", "/predict/v2", dict(v2_request, candidates=[v2_candidate(1, same_members * 21, 0.9)]))
check(status == 422, "/predict/v2 rejects more than 20 members")
status, _ = call("POST", "/predict/v2", dict(v2_request, candidates=[v2_candidate(1, [], 0.9)]))
check(status == 422, "/predict/v2 rejects a candidate without members")

items = [{"title": f"Story number {i} about the {word} today", "description": None}
         for i, word in enumerate(["cricket", "monsoon", "election", "volcano", "market"] * 24)]
status, _ = call("POST", "/vocabulary/v2", {"items": items[:10]})
check(status == 422, "/vocabulary/v2 rejects fewer than 100 items")
status, body = call("POST", "/vocabulary/v2", {"items": items})
check(status == 200 and body["documents"] == 120, "/vocabulary/v2 refits on 120 items")
status, after = call("POST", "/predict/v2", v2_request)
check(status == 200 and after["vocabulary_version"] == body["vocabulary_version"], "/predict/v2 uses the refitted vocabulary")

status, body = call("POST", "/entities", {"title": "Supreme Court quashes detention of Mulla Afroz",
                                        "description": "India face West Indies in the second ODI."})
check(status == 200 and body["extractor_version"].startswith("rules"), "/entities returns 200 with a version")
found = {(m["normalized"], m["type"]) for m in body["mentions"]}
check({("supreme court", "name"), ("mulla afroz", "name"), ("india", "team")} <= found, "/entities finds names and teams")
status, _ = call("POST", "/entities", {"title": "x"}, api_key="wrong-key")
check(status == 401, "/entities rejects a wrong API key with 401")

print(f"PASS (v1 same={v1_scores[1]:.3f}, unrelated={v1_scores[2]:.3f}; "
      f"v2 same={v2_scores[1]:.3f}, unrelated={v2_scores[2]:.3f})")
