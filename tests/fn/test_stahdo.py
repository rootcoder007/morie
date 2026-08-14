"""Stahel-Donoho outlyingness and estimator."""
import importlib
import math

import pytest

S = importlib.import_module("morie.fn.stahdo")

JIT = [0.21, -0.34, 0.12, -0.18, 0.29, -0.07, 0.33, -0.25, 0.16,
       -0.31, 0.08]
CLEAN = [[float(t), t + JIT[i]] for i, t in enumerate(range(11))]
HIDDEN = CLEAN + [[3.0, 8.0]]
A = [[2.0, 1.0], [-1.0, 3.0]]
B = [5.0, -2.0]
TX = [[A[0][0] * r[0] + A[0][1] * r[1] + B[0],
       A[1][0] * r[0] + A[1][1] * r[1] + B[1]] for r in HIDDEN]


def test_one_dimension_is_the_closed_form():
    v = [1.0, 2.0, 3.0, 4.0, 100.0]
    m, s = S.median(v), S.mad(v)
    o = S.outlyingness([[x] for x in v])["outlyingness"]
    for i, x in enumerate(v):
        assert o[i] == pytest.approx(abs(x - m) / s, abs=1e-12)


def test_the_mad_is_normal_consistent():
    assert S.mad([0.0, 1, 2, 3, 4]) == pytest.approx(1.4826)
    assert S.mad([0.0, 1, 2, 3, 4], consistent=False) \
        == pytest.approx(1.0)


def test_the_chi_square_median_at_two_df():
    assert S._chi2_median(2) == pytest.approx(2.0 * math.log(2.0),
                                              abs=1e-9)


def test_it_finds_an_outlier_hidden_in_every_margin():
    marg = [abs(HIDDEN[-1][k] - S.median([r[k] for r in HIDDEN]))
            / S.mad([r[k] for r in HIDDEN]) for k in (0, 1)]
    o = S.outlyingness(HIDDEN)["outlyingness"]
    assert max(marg) < 2.0
    assert o[-1] > 5.0 * max(o[:-1])


def test_the_subsample_estimator_is_affine_equivariant():
    f0, f1 = (S.stahel_donoho(HIDDEN, "subsample"),
              S.stahel_donoho(TX, "subsample"))
    want = [A[a][0] * f0["location"][0] + A[a][1] * f0["location"][1]
            + B[a] for a in (0, 1)]
    assert all(f1["location"][k] == pytest.approx(want[k], abs=1e-8)
               for k in (0, 1))
    for a in (0, 1):
        for b in (0, 1):
            expect = sum(A[a][i] * f0["scatter"][i][j] * A[b][j]
                         for i in (0, 1) for j in (0, 1))
            assert f1["scatter"][a][b] == pytest.approx(expect,
                                                        abs=1e-6)


def test_the_outlyingness_is_affine_invariant():
    a = S.outlyingness(HIDDEN)["outlyingness"]
    b = S.outlyingness(TX)["outlyingness"]
    assert all(x == pytest.approx(y, abs=1e-8) for x, y in zip(a, b))


def test_the_random_route_is_not_equivariant():
    g0 = S.stahel_donoho(HIDDEN, "random", 400, 3)
    g1 = S.stahel_donoho(TX, "random", 400, 3)
    want = [A[a][0] * g0["location"][0] + A[a][1] * g0["location"][1]
            + B[a] for a in (0, 1)]
    assert max(abs(g1["location"][k] - want[k]) for k in (0, 1)) \
        > 1e-6


def test_it_resists_thirty_per_cent_contamination():
    good = [[t * 0.5, t * 0.5 + 1.0] for t in range(14)]
    bad = good + [[60.0, -60.0]] * 6
    clean = [sum(r[k] for r in good) / len(good) for k in (0, 1)]
    mean = [sum(r[k] for r in bad) / len(bad) for k in (0, 1)]
    rob = S.stahel_donoho(bad)["location"]
    assert all(abs(rob[k] - clean[k]) < 1.0 for k in (0, 1))
    assert max(abs(mean[k] - clean[k]) for k in (0, 1)) > 10.0
    assert S.stahel_donoho(bad)["n_downweighted"] >= 6


def test_a_small_problem_exhausts_the_subsample_family():
    r = S.stahel_donoho(HIDDEN)
    assert r["exhaustive"]
    assert r["n_directions"] == 66


def test_the_weight_function():
    r = S.stahel_donoho(HIDDEN)
    for w, x in zip(r["weights"], r["outlyingness"]):
        if x <= r["cutoff"]:
            assert w == 1.0
        else:
            assert w == pytest.approx((r["cutoff"] / x) ** 2)


def test_the_scatter_is_symmetric():
    r = S.stahel_donoho(HIDDEN)
    assert r["scatter"][0][1] == pytest.approx(r["scatter"][1][0])
    assert r["n"] == 12 and r["p"] == 2


@pytest.mark.parametrize("call", [
    lambda: S.outlyingness([[1.0], [2.0]]),
    lambda: S.outlyingness([[1.0, 2.0], [1.0]]),
    lambda: S.outlyingness(HIDDEN, "grid"),
    lambda: S.median([]),
    lambda: S.stahel_donoho(HIDDEN, cutoff=0.0),
    lambda: S.outlyingness([[1.0], [1.0], [1.0], [1.0]]),
])
def test_bad_input_is_refused(call):
    with pytest.raises(ValueError):
        call()
