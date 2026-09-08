"""Whole-genome regression: two-level ridge with LOCO."""
import importlib
import math

import pytest

R = importlib.import_module("morie.fn.regmlm")
np = importlib.import_module("morie.fn._array_core")
Rng = importlib.import_module("morie.fn.survrsf")._Rng


def sim(n=100, m=40, seed=3):
    rng = Rng(seed)
    chrom = [c for c in range(4) for _ in range(m // 4)]
    G = [[1.0 if rng.next() < 0.5 else 0.0 for _ in range(m)]
         for _ in range(n)]
    y = [sum(G[i][10:14]) * 0.8
         + (rng.next() + rng.next() + rng.next() - 1.5)
         for i in range(n)]
    return G, y, chrom


G, Y, CHROM = sim()


def test_blocks_tile_the_markers_exactly_once():
    b = R.make_blocks(2500, [0] * 1000 + [1] * 1500, 1000)
    assert b[0]["start"] == 0
    assert b[-1]["stop"] == 2500
    assert all(b[i]["stop"] == b[i + 1]["start"]
               for i in range(len(b) - 1))


def test_blocks_never_straddle_a_chromosome():
    b = R.make_blocks(2500, [0] * 1000 + [1] * 1500, 1000)
    assert all(x["stop"] <= 1000 or x["start"] >= 1000 for x in b)


def test_the_papers_reduction():
    b = R.make_blocks(500000, None, 1000)
    assert len(b) == 500
    assert len(b) * 5 == 2500


@pytest.mark.parametrize("bad", [
    lambda: R.make_blocks(0),
    lambda: R.make_blocks(10, block_size=0),
    lambda: R.make_blocks(10, [0] * 5),
])
def test_invalid_block_requests_are_refused(bad):
    with pytest.raises(ValueError):
        bad()


def test_ridge_at_zero_is_least_squares():
    rng = Rng(3)
    n, p = 50, 3
    X = [[rng.next() for _ in range(p)] for _ in range(n)]
    y = [sum(x) for x in X]
    r = R.ridge_fit(X, y, 0.0)
    A = [[sum(X[i][a] * X[i][c] for i in range(n)) for c in range(p)]
         for a in range(p)]
    b = [sum(X[i][a] * y[i] for i in range(n)) for a in range(p)]
    ols = [float(v) for v in np.linalg.solve(np.array(A), np.array(b))]
    assert r["beta"] == pytest.approx(ols, abs=1e-9)


def test_ridge_shrinks_monotonically():
    rng = Rng(4)
    X = [[rng.next() for _ in range(3)] for _ in range(40)]
    y = [sum(x) for x in X]
    norms = [sum(v * v for v in R.ridge_fit(X, y, lam)["beta"])
             for lam in (0.0, 1.0, 100.0, 1e6)]
    assert all(norms[i] > norms[i + 1] for i in range(3))


def test_a_negative_shrinkage_is_refused():
    with pytest.raises(ValueError):
        R.ridge_fit([[1.0]], [1.0], -1.0)


def test_loco_excludes_its_own_chromosome():
    f = R.fit(G, Y, CHROM, block_size=5, n_ridge=3, k=4)
    meta = f["level0"]["meta"]
    w = f["level1"]["weights"]
    for c in f["chromosomes"]:
        rebuilt = [sum(f["level0"]["predictors"][j][i] * w[j]
                       for j in range(len(meta))
                       if meta[j]["chromosome"] != c)
                   for i in range(len(Y))]
        assert rebuilt == pytest.approx(f["loco"][c], abs=1e-12)


def test_backgrounds_differ_between_chromosomes():
    f = R.fit(G, Y, CHROM, block_size=5, n_ridge=3, k=4)
    assert max(abs(f["loco"][0][i] - f["loco"][1][i])
               for i in range(len(Y))) > 1e-9


def test_proximal_contamination_is_visible():
    f = R.fit(G, Y, CHROM, block_size=5, n_ridge=3, k=4)
    g = [G[i][11] for i in range(len(Y))]
    loco = R.test_variant(g, Y, f["loco"][1])
    full = R.test_variant(g, Y, f["level1"]["prediction"])
    assert loco["chisq"] > full["chisq"]


def test_the_planted_variant_is_detected():
    f = R.fit(G, Y, CHROM, block_size=5, n_ridge=3, k=4)
    g = [G[i][11] for i in range(len(Y))]
    assert R.test_variant(g, Y, f["loco"][1])["p_value"] < 1e-3


def test_a_monomorphic_variant_is_refused():
    f = R.fit(G, Y, CHROM, block_size=5, n_ridge=3, k=4)
    with pytest.raises(ValueError):
        R.test_variant([1.0] * len(Y), Y, f["loco"][0])


def test_offset_length_is_checked():
    with pytest.raises(ValueError):
        R.test_variant([1.0, 0.0], [1.0, 2.0], [0.0])


@pytest.mark.parametrize("cv", R.CV_SCHEMES)
def test_both_cv_schemes_run(cv):
    f = R.fit(G, Y, CHROM, block_size=5, n_ridge=2, cv=cv, k=4)
    assert len(f["level1"]["out_of_fold"]) == len(Y)


def test_an_unknown_cv_scheme_is_refused():
    with pytest.raises(ValueError):
        R.level1_stack([[1.0, 2.0]], [1.0, 2.0], cv="jackknife")


def test_out_of_fold_differs_from_in_sample():
    f = R.fit(G, Y, CHROM, block_size=5, n_ridge=3, k=4)
    l1 = f["level1"]
    assert any(abs(l1["out_of_fold"][i] - l1["prediction"][i]) > 1e-9
               for i in range(len(Y)))


def test_mismatched_inputs_are_refused():
    with pytest.raises(ValueError):
        R.fit(G, Y[:10], CHROM)
