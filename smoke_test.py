"""Smoke test for a running AI service.

Checks health, API-key protection, /embed and /predict end to end, so a
broken model file, dependency or endpoint fails CI instead of production.

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

print(f"PASS (same={scores[1]:.3f}, unrelated={scores[2]:.3f})")
