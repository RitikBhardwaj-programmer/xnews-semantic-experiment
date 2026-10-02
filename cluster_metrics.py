"""
Clustering metrics for event grouping (V4 approach 1, stage 0).

All functions take two equal-length label sequences: the TRUE event of each
article and the PREDICTED event of each article. Labels can be any hashable
values; only "same label or not" matters.

- B-cubed (primary): per-article precision/recall of its predicted group vs.
  its true group, averaged over articles (Bagga & Baldwin 1998; the only
  family meeting all constraints in Amigo et al. 2009).
- Pairwise: precision/recall over article pairs placed together.
- CEAF-e: one-to-one alignment of predicted and true events (Luo 2005), so
  errors on small events count as much as on big ones.
- ARI / AMI: chance-corrected agreement (scikit-learn).

Over-merging lowers the precision-type scores; over-splitting lowers recall.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    pair_confusion_matrix,
)


def _codes(labels):
    return pd.factorize(pd.Series(list(labels)))[0]


def f1(precision, recall):
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def bcubed_items(true, pred):
    """Per-article B-cubed precision and recall (arrays)."""

    t, p = _codes(true), _codes(pred)

    overlap = pd.Series(1, index=pd.MultiIndex.from_arrays([t, p])).groupby(level=[0, 1]).transform("size").to_numpy()
    pred_size = np.bincount(p)[p]
    true_size = np.bincount(t)[t]

    return overlap / pred_size, overlap / true_size


def bcubed(true, pred):

    precision, recall = (float(x.mean()) for x in bcubed_items(true, pred))

    return precision, recall, f1(precision, recall)


def pairwise(true, pred):

    (_, fp), (fn, tp) = pair_confusion_matrix(_codes(true), _codes(pred))

    precision = 1.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 1.0 if tp + fn == 0 else tp / (tp + fn)

    return float(precision), float(recall), f1(float(precision), float(recall))


def ceaf_e(true, pred):

    t, p = _codes(true), _codes(pred)

    overlap = np.zeros((t.max() + 1, p.max() + 1))
    np.add.at(overlap, (t, p), 1)

    t_size = overlap.sum(axis=1, keepdims=True)
    p_size = overlap.sum(axis=0, keepdims=True)
    similarity = 2 * overlap / (t_size + p_size)

    rows, cols = linear_sum_assignment(similarity, maximize=True)
    total = similarity[rows, cols].sum()

    precision = total / overlap.shape[1]
    recall = total / overlap.shape[0]

    return float(precision), float(recall), f1(float(precision), float(recall))


def summary(true, pred):
    """One row of every metric plus size statistics."""

    t, p = _codes(true), _codes(pred)
    pred_sizes = np.sort(np.bincount(p))[::-1]
    true_sizes = np.sort(np.bincount(t))[::-1]

    b = bcubed(t, p)
    w = pairwise(t, p)
    c = ceaf_e(t, p)

    return {
        "bcubed_p": b[0], "bcubed_r": b[1], "bcubed_f1": b[2],
        "pair_p": w[0], "pair_r": w[1], "pair_f1": w[2],
        "ceafe_p": c[0], "ceafe_r": c[1], "ceafe_f1": c[2],
        "ari": float(adjusted_rand_score(t, p)),
        "ami": float(adjusted_mutual_info_score(t, p)),
        "events_pred": len(pred_sizes),
        "events_true": len(true_sizes),
        "singletons_pred": float((pred_sizes == 1).mean()),
        "largest_pred": pred_sizes[:5].tolist(),
        "largest_true": true_sizes[:5].tolist(),
    }


def pair_components(true, pred):
    """Per TRUE event: same-event pairs kept together (tp), split apart (fn),
    and wrong pairs it takes part in (fp, half of each cross-event pair)."""

    t, p = _codes(true), _codes(pred)
    joint = pd.crosstab(t, p).to_numpy()
    true_sizes = joint.sum(axis=1)
    pred_sizes = joint.sum(axis=0)
    tp = (joint * (joint - 1) / 2).sum(axis=1)
    fn = true_sizes * (true_sizes - 1) / 2 - tp
    # pairs that share a predicted event but not a true event, split between the two events
    fp = (joint * (pred_sizes - joint)).sum(axis=1) / 2
    return tp, fn, fp


def bootstrap_pairwise(true, preds, n=1000, seed=7):
    """Cluster bootstrap of pairwise F1 (resampling true events), paired across systems."""

    components = {name: pair_components(true, pred) for name, pred in preds.items()}
    events = len(next(iter(components.values()))[0])
    rng = np.random.default_rng(seed)
    scores = {name: np.empty(n) for name in preds}

    for k in range(n):
        idx = rng.integers(0, events, events)
        for name, (tp, fn, fp) in components.items():
            t, f_n, f_p = tp[idx].sum(), fn[idx].sum(), fp[idx].sum()
            precision = t / (t + f_p) if t + f_p else 1.0
            recall = t / (t + f_n) if t + f_n else 1.0
            scores[name][k] = f1(precision, recall)

    interval = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
    names = list(preds)
    return {
        "f1": {name: interval(scores[name]) for name in names},
        "diff": {(a, b): interval(scores[a] - scores[b]) for i, a in enumerate(names) for b in names[i + 1:]},
    }


def bootstrap_bcubed(true, preds, groups=None, n=1000, seed=7):
    """Cluster bootstrap of B-cubed F1.

    Resamples whole TRUE events (or the given groups, e.g. days) with
    replacement and re-averages the per-article B-cubed scores computed on
    the full data. `preds` maps system name -> predicted labels. Returns,
    per system, the 95% interval of F1, and for every pair of systems the
    95% interval of the F1 difference (paired: the same resamples).
    """

    groups = _codes(true if groups is None else groups)
    members = [np.flatnonzero(groups == g) for g in range(groups.max() + 1)]
    items = {name: bcubed_items(true, pred) for name, pred in preds.items()}

    rng = np.random.default_rng(seed)
    scores = {name: np.empty(n) for name in preds}

    for k in range(n):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        for name, (precision, recall) in items.items():
            scores[name][k] = f1(precision[idx].mean(), recall[idx].mean())

    interval = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
    names = list(preds)

    return {
        "f1": {name: interval(scores[name]) for name in names},
        "diff": {
            (a, b): interval(scores[a] - scores[b])
            for i, a in enumerate(names) for b in names[i + 1:]
        },
    }
