"""Tests for hntfst -- honest random forests.

Honesty is structural, so it is tested structurally: permute the
I-sample responses and require the tree not to move. Full anchor:
ledger/wave3/anchor_hntfst.py.
"""
import math
import pytest
from morie.fn import _array_core as np
from morie.fn import _s03core as k
from morie.fn.hntfst import (honest_forest, honest_tree,
                             infinitesimal_jackknife, tree_predict)


def structure(t):
    if t["leaf"]:
        return ["leaf"]
    return ([(t["feature"], round(t["threshold"], 12))]
            + structure(t["left"]) + structure(t["right"]))


def draw(n, seed):
    rng = np.random.default_rng(seed)
    X = [[rng.standard_normal() for _ in range(4)] for _ in range(n)]
    mu = [1.5 * X[i][0] - 1.0 * X[i][1] + 0.5 * X[i][0] * X[i][1]
          for i in range(n)]
    y = [mu[i] + 0.4 * rng.standard_normal() for i in range(n)]
    W = [1.0 if float(rng.uniform()) < 0.5 else 0.0 for _ in range(n)]
    return {"X": X, "y": y, "mu": mu, "W": W, "n": n}


@pytest.fixture(scope="module")
def data():
    return draw(400, 5)


def _permute(y, idx, seed):
    rng = np.random.default_rng(seed)
    out = list(y)
    for a, b in zip(idx, sorted(idx, key=lambda _i: float(rng.uniform()))):
        out[a] = y[b]
    return out


def test_procedure_one_splits_into_disjoint_halves(data):
    _, info = honest_tree(data["X"], data["y"], min_leaf=5, seed=11)
    assert not (set(info["I"]) & set(info["J"]))
    assert len(info["I"]) + len(info["J"]) == len(info["subsample"])


def test_honesty_the_splits_ignore_the_i_sample_responses(data):
    """The check that can actually fail: an implementation quietly using
    the I responses would shift, and no accuracy comparison would show
    it."""
    tree, info = honest_tree(data["X"], data["y"], min_leaf=5, seed=11)
    tp, _ = honest_tree(data["X"], _permute(data["y"], info["I"], 99),
                        min_leaf=5, seed=11)
    assert structure(tree) == structure(tp)
    # but the leaf values move, so the permutation was real
    assert any(abs(tree_predict(tree, data["X"][i])
                   - tree_predict(tp, data["X"][i])) > 1e-9
               for i in range(data["n"]))
    # and permuting J's responses MUST move the splits
    tj, _ = honest_tree(data["X"], _permute(data["y"], info["J"], 98),
                        min_leaf=5, seed=11)
    assert structure(tree) != structure(tj)


def test_the_adaptive_tree_fails_the_honesty_test(data):
    """The control: without it, the test above would pass for a tree
    that ignored every response."""
    ta, ia = honest_tree(data["X"], data["y"], kind="adaptive",
                         min_leaf=5, seed=11)
    ta2, _ = honest_tree(data["X"], _permute(data["y"], ia["I"], 97),
                         kind="adaptive", min_leaf=5, seed=11)
    assert structure(ta) != structure(ta2)


def test_the_propensity_tree_ignores_y_entirely(data):
    tp, ip = honest_tree(data["X"], data["y"], W=data["W"],
                         kind="propensity", min_leaf=5, seed=7)
    rng = np.random.default_rng(3)
    yshuf = sorted(data["y"], key=lambda _v: float(rng.uniform()))
    tp2, _ = honest_tree(data["X"], yshuf, W=data["W"],
                         kind="propensity", min_leaf=5, seed=7)
    assert structure(tp) == structure(tp2)
    assert set(ip["I"]) == set(ip["J"]) == set(ip["subsample"])
    with pytest.raises(ValueError):
        honest_tree(data["X"], data["y"], kind="propensity")


def test_definition_three_every_feature_gets_split_on(data):
    f = honest_forest(data["X"], data["y"], n_trees=120, min_leaf=5,
                      pi=0.5, seed=3)
    assert all(v > 0.02 for v in f["split_share"])
    # and the predictive features are still preferred
    assert (f["split_share"][0] + f["split_share"][1]
            > f["split_share"][2] + f["split_share"][3])


def test_honesty_prevents_inventing_structure_from_noise():
    """The concrete thing honesty buys: on an outcome with NO signal,
    the adaptive forest explains a third of it and the honest one
    does not."""
    r2 = {}
    for kind in ("double-sample", "adaptive"):
        vals = []
        for rep in range(5):
            rng = np.random.default_rng(500 + rep)
            X = [[rng.standard_normal() for _ in range(4)]
                 for _ in range(400)]
            y = [rng.standard_normal() for _ in range(400)]
            r = honest_forest(X, y, kind=kind, n_trees=100, min_leaf=5,
                              seed=rep)
            ybar = k.mean(y)
            ss = sum((y[i] - ybar) ** 2 for i in range(400))
            rss = sum((y[i] - r["fitted"][i]) ** 2 for i in range(400))
            vals.append(1.0 - rss / ss)
        r2[kind] = k.mean(vals)
    assert r2["adaptive"] > 0.2
    assert r2["double-sample"] < 0.1


def test_the_ij_correction_is_the_papers_factor(data):
    at = [[0.5, -0.5, 0.0, 0.0]]
    on = honest_forest(data["X"], data["y"], n_trees=150, min_leaf=5,
                       seed=4, at=at, correction=True)
    off = honest_forest(data["X"], data["y"], n_trees=150, min_leaf=5,
                        seed=4, at=at, correction=False)
    n, s = on["n"], on["s"]
    want = (n - 1.0) / n * (float(n) / (n - s)) ** 2
    assert on["variance"][0] / off["variance"][0] == pytest.approx(
        want, abs=1e-9)
    assert on["fitted"][0] == pytest.approx(off["fitted"][0], abs=1e-15)
    assert all(v >= 0.0 for v in on["variance"])


def test_the_ij_is_zero_when_every_tree_agrees():
    assert infinitesimal_jackknife([2.0] * 10,
                                   [[True] * 20 for _ in range(10)],
                                   20, 10) == 0.0


def test_error_falls_as_n_grows():
    truth = 1.5 * 0.5 - 1.0 * -0.5 + 0.5 * 0.5 * -0.5
    errs = []
    for n in (200, 800):
        d = draw(n, 77)
        r = honest_forest(d["X"], d["y"], n_trees=100, min_leaf=5,
                          seed=1, at=[[0.5, -0.5, 0.0, 0.0]])
        errs.append(abs(r["fitted"][0] - truth))
    assert errs[-1] < errs[0]


def test_argument_checks(data):
    with pytest.raises(ValueError):
        honest_forest(data["X"], data["y"], kind="nope")
    with pytest.raises(ValueError):
        honest_tree(data["X"], data["y"], alpha=0.7)
    with pytest.raises(ValueError):
        honest_tree(data["X"], data["y"], pi=0.0)
    with pytest.raises(ValueError):
        honest_forest(data["X"], data["y"], subsample_frac=1.5)
    with pytest.raises(ValueError):
        honest_forest(data["X"], data["y"], n_trees=1)
    with pytest.raises(ValueError):
        honest_forest(data["X"][:-1], data["y"])
    with pytest.raises(ValueError):
        honest_forest(data["X"][:10], data["y"][:10])
    with pytest.raises(ValueError):
        infinitesimal_jackknife([1.0], [[True]], 5, 2)
    with pytest.raises(ValueError):
        infinitesimal_jackknife([1.0, 2.0], [[True] * 5] * 2, 5, 5)
