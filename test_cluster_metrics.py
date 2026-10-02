"""
Checks cluster_metrics against small hand-computed cases.
Run: python test_cluster_metrics.py
"""

from cluster_metrics import bcubed, bootstrap_bcubed, bootstrap_pairwise, ceaf_e, pair_components, pairwise, summary

TRUE = ["a", "a", "a", "b", "b", "c"]


def close(x, y, tol=1e-9):
    return abs(x - y) <= tol


def test_perfect():
    for metric in (bcubed, pairwise, ceaf_e):
        assert metric(TRUE, [1, 1, 1, 2, 2, 3]) == (1.0, 1.0, 1.0), metric.__name__


def test_label_names_do_not_matter():
    assert bcubed(TRUE, ["x", "x", "x", "y", "y", "z"]) == bcubed(TRUE, TRUE)


def test_everything_merged():
    # One predicted group of 6. B-cubed precision per article = |own true group| / 6.
    p, r, _ = bcubed(TRUE, [0] * 6)
    assert close(p, (3 * 3 / 6 + 2 * 2 / 6 + 1 / 6) / 6)   # 14/36
    assert close(r, 1.0)
    # Pairs: 15 predicted together, 3 + 1 truly together.
    p, r, _ = pairwise(TRUE, [0] * 6)
    assert close(p, 4 / 15) and close(r, 1.0)


def test_everything_split():
    p, r, _ = bcubed(TRUE, range(6))
    assert close(p, 1.0)
    assert close(r, (3 * 1 / 3 + 2 * 1 / 2 + 1) / 6)       # 3/6
    p, r, _ = pairwise(TRUE, range(6))
    assert close(p, 1.0) and close(r, 0.0)


def test_one_wrong_merge():
    # true b is merged into a: predicted groups {a,a,a,b,b}, {c}.
    pred = [0, 0, 0, 0, 0, 1]
    p, r, _ = bcubed(TRUE, pred)
    assert close(p, (3 * 3 / 5 + 2 * 2 / 5 + 1) / 6)       # 18/30
    assert close(r, 1.0)
    # CEAF-e: best alignment a<->{5 items}: 2*3/(3+5) = 0.75, c<->{c}: 1, b unmatched.
    p, r, _ = ceaf_e(TRUE, pred)
    assert close(p, 1.75 / 2) and close(r, 1.75 / 3)


def test_summary_sizes():
    row = summary(TRUE, [0, 0, 0, 0, 0, 1])
    assert row["events_pred"] == 2 and row["events_true"] == 3
    assert row["largest_pred"] == [5, 1] and row["largest_true"] == [3, 2, 1]


def test_bootstrap_identical_systems_have_zero_difference():
    result = bootstrap_bcubed(TRUE, {"x": TRUE, "y": TRUE}, n=50)
    assert result["diff"][("x", "y")] == (0.0, 0.0)
    assert result["f1"]["x"] == (1.0, 1.0)


def test_pair_components_add_up_to_pairwise_counts():
    pred = [0, 0, 0, 0, 0, 1]                  # true b merged into a
    tp, fn, fp = pair_components(TRUE, pred)
    assert tp.sum() == 4 and fn.sum() == 0     # a: 3 pairs, b: 1 pair, all kept together
    assert fp.sum() == 6                       # 3 x 2 a-b pairs wrongly together
    p, r, _ = pairwise(TRUE, pred)
    assert close(p, tp.sum() / (tp.sum() + fp.sum())) and close(r, 1.0)


def test_bootstrap_pairwise_identical_systems():
    result = bootstrap_pairwise(TRUE, {"x": [0, 0, 0, 1, 1, 2], "y": [0, 0, 0, 1, 1, 2]}, n=50)
    assert result["diff"][("x", "y")] == (0.0, 0.0)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} tests passed")
