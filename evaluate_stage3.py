"""
Stage 3 of V4 approach 1 (offline): does a different 384-dimension embedder
beat all-MiniLM-L6-v2 inside the shipped v2 matcher?

Candidates are same-size drop-ins (384 dimensions, so the pgvector column
and index stay as they are): BAAI/bge-small-en-v1.5 and thenlper/gte-small.
Each embeds the same text production embeds (title + "\\n" + description).
MiniLM uses the embeddings stored by production, so it reproduces the
shipped result. Everything else is the shipped v2 setup, same folds:
`evaluate_stage1.refit_fold(..., "B nightly refit + cap 20, no temporal", day)`.

Criterion (fixed before the run): a candidate is adopted only if pairwise F1
improves on MiniLM with a 95% paired cluster-bootstrap interval above 0,
B-cubed F1 is not worse (point estimate), and mixed events do not increase.

Embeddings are cached in data/validation/eval_days/local/ (git-ignored).

Usage:
    python evaluate_stage3.py
"""

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from cluster_metrics import bootstrap_bcubed, bootstrap_pairwise, summary
from evaluate_stage1 import BASE, DAYS, load, refit_fold

SHIPPED = "B nightly refit + cap 20, no temporal"
CANDIDATES = ["BAAI/bge-small-en-v1.5", "thenlper/gte-small"]
BASELINE = "all-MiniLM-L6-v2 (production embeddings)"


def embeddings_for(name, frames):
    """Cached {article id: unit vector} for one embedder."""
    cache = BASE / "local" / f"embeddings_{name.replace('/', '__')}.npz"
    if cache.exists():
        data = np.load(cache)
        return dict(zip(data["ids"], data["vectors"]))
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(name)
    allf = pd.concat(frames)
    texts = (allf["title"] + "\n" + allf["description"].fillna("")).tolist()
    vectors = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    assert vectors.shape[1] == 384, f"{name} gives {vectors.shape[1]} dimensions, not 384"
    np.savez(cache, ids=allf["id"].to_numpy(), vectors=vectors)
    return dict(zip(allf["id"], vectors))


def with_vectors(frames, vectors):
    return [f.assign(vector=[np.asarray(vectors[i], dtype=np.float64) for i in f["id"]]) for f in frames]


def mixed_events(truth, pred):
    frame = pd.DataFrame({"t": truth, "p": pred})
    return int((frame.groupby("p")["t"].nunique() >= 2).sum())


def main():
    started = time.time()
    frames = [load(d) for d in DAYS]
    systems = {BASELINE: frames}
    for name in CANDIDATES:
        systems[name] = with_vectors(frames, embeddings_for(name, frames))
        print(f"embedded with {name} ({time.time() - started:.0f}s)", flush=True)

    cells = [(name, test) for name in systems for test in range(len(DAYS))]
    outputs = Parallel(n_jobs=len(cells))(
        delayed(refit_fold)(systems[name], SHIPPED, test) for name, test in cells)
    print(f"replays done ({time.time() - started:.0f}s)", flush=True)

    results = {name: [None] * len(DAYS) for name in systems}
    for (name, test), (assigned, threshold) in zip(cells, outputs):
        results[name][test] = assigned
        print(f"  {name}: fold {DAYS[test]} threshold {threshold}")

    truth = np.concatenate([[f"{d}:{x}" for x in f["truth"]] for d, f in zip(DAYS, frames)])
    preds = {n: np.concatenate([[f"{d}:{x}" for x in p] for d, p in zip(DAYS, ps)]) for n, ps in results.items()}
    table = pd.DataFrame({n: summary(truth, p) for n, p in preds.items()}).T
    table["mixed"] = [mixed_events(truth, p) for p in preds.values()]
    with pd.option_context("display.width", 220, "display.float_format", lambda x: f"{x:.3f}"):
        print(table[["bcubed_p", "bcubed_r", "bcubed_f1", "pair_p", "pair_r", "pair_f1",
                     "events_pred", "mixed", "largest_pred"]].to_string())

    pairwise_ci = bootstrap_pairwise(truth, preds)
    bcubed_ci = bootstrap_bcubed(truth, preds)
    print("\n== criterion (vs MiniLM; 95% paired cluster-bootstrap interval of the difference)")
    for name in CANDIDATES:
        pw = pairwise_ci["diff"][(BASELINE, name)]
        bc = bcubed_ci["diff"][(BASELINE, name)]
        pw, bc = (-pw[1], -pw[0]), (-bc[1], -bc[0])
        ok = (pw[0] > 0 and table.loc[name, "bcubed_f1"] >= table.loc[BASELINE, "bcubed_f1"]
              and table.loc[name, "mixed"] <= table.loc[BASELINE, "mixed"])
        print(f"  {name}: pairwise F1 diff {pw[0]:+.3f}..{pw[1]:+.3f}, B-cubed F1 diff {bc[0]:+.3f}..{bc[1]:+.3f}, "
              f"mixed {table.loc[name, 'mixed']} vs {table.loc[BASELINE, 'mixed']} -> {'ADOPT' if ok else 'KEEP MiniLM'}")
    print(f"({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
