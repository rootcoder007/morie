"""Tests for andmnh (Andrews & Monahan 1992, VAR prewhitened kernel HAC)."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn._rng import random_normal
from morie.fn.andmnh import (KERNEL_CONSTANTS, KERNELS, alpha_ar1, ar1_fit,
                             andrews_monahan_hac, automatic_bandwidth,
                             kernel_hac, moment_vectors, prewhiten_var,
                             singular_value_adjust)


def _kq_numeric(kfun, q):
    """(1 - k(x))/|x|^q as x -> 0, Richardson-extrapolated."""
    a = (1.0 - kfun(5e-3)) / 5e-3 ** q
    b = (1.0 - kfun(2.5e-3)) / 2.5e-3 ** q
    return b + (b - a) / 3.0


def _int_k2(kfun, hi, n=40001):
    step = hi / (n - 1)
    tot = 0.0
    for i in range(n):
        w = 1.0 if i in (0, n - 1) else (4.0 if i % 2 else 2.0)
        tot += w * kfun(i * step) ** 2
    return 2.0 * tot * step / 3.0


# --------------------------------------------------------------------------
# the kernels and their constants
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(KERNELS))
def test_kernel_constants_come_from_the_kernel_itself(name):
    """Nothing here may rest on a transcribed number."""
    kfun = KERNELS[name]
    q, kq, ik2, bounded = KERNEL_CONSTANTS[name]
    assert kfun(0.0) == pytest.approx(1.0)
    assert _kq_numeric(kfun, q) == pytest.approx(kq, rel=3e-3)
    assert _int_k2(kfun, 1.0 if bounded else 300.0) == pytest.approx(
        ik2, abs=5e-4)
    if bounded:
        assert kfun(1.0001) == 0.0
    # k is even
    for x in (0.13, 0.7, 1.4):
        assert kfun(-x) == pytest.approx(kfun(x), abs=1e-15)


def test_qs_kernel_is_the_printed_equation_3_2():
    x = 0.37
    z = 6.0 * math.pi * x / 5.0
    want = (25.0 / (12.0 * math.pi ** 2 * x * x)) * (math.sin(z) / z
                                                     - math.cos(z))
    assert KERNELS["qs"](x) == pytest.approx(want, rel=1e-14)
    # unbounded support: it is still non-zero well past 1
    assert abs(KERNELS["qs"](4.3)) > 1e-4


def test_equation_3_5_falls_out_of_the_general_bandwidth_formula():
    q, kq, ik2, _ = KERNEL_CONSTANTS["qs"]
    assert (q * kq * kq / ik2) ** 0.2 == pytest.approx(1.3221, abs=5e-5)
    v = [[float(x)] for x in random_normal(400, seed=4, stream=0)]
    s, alpha, _ = automatic_bandwidth(v, kernel="qs")
    assert s == pytest.approx(1.3221 * (alpha * 400) ** 0.2, abs=1e-4)


# --------------------------------------------------------------------------
# footnote 4
# --------------------------------------------------------------------------

def test_svd_adjustment_keeps_i_minus_a_away_from_singular():
    big = [[1.4, 0.3, -0.2], [0.1, 1.1, 0.5], [-0.6, 0.2, 0.9]]
    adj = singular_value_adjust(big, 0.97)
    assert max(float(v) for v in np.linalg.svd(np.asarray(adj))[1]) \
        <= 0.97 + 1e-12
    ev = [abs(complex(v))
          for v in np.linalg.eigvals(np.eye(3) - np.asarray(adj))]
    assert min(ev) >= 0.03 - 1e-12


def test_a_matrix_inside_the_cap_is_left_alone():
    small = [[0.4, 0.1], [0.0, -0.3]]
    got = singular_value_adjust(small, 0.97)
    for i in range(2):
        for j in range(2):
            assert float(got[i][j]) == pytest.approx(small[i][j], abs=1e-12)


def test_singular_value_adjust_rejects_a_bad_cap():
    for bad in (0.0, 1.0, 1.5, -0.2):
        with pytest.raises(ValueError):
            singular_value_adjust([[0.1]], bad)


# --------------------------------------------------------------------------
# equation 2.3
# --------------------------------------------------------------------------

_Z = [float(x) for x in random_normal(200, seed=11, stream=1)]
_V = [[_Z[2 * t], 0.5 * _Z[2 * t] + _Z[2 * t + 1]] for t in range(100)]


@pytest.mark.parametrize("kern", sorted(KERNELS))
@pytest.mark.parametrize("S", [2.5, 7.0])
def test_equation_2_3_against_an_independent_double_sum(kern, S):
    """sum_j k(j/S) Gamma(j) == (1/T) sum_s sum_t k((t-s)/S) V_t V_s'."""
    got = kernel_hac(_V, S, kernel=kern, n_params=2)
    kfun = KERNELS[kern]
    T = len(_V)
    ref = [[0.0, 0.0], [0.0, 0.0]]
    for a in range(T):
        for b in range(T):
            w = kfun((a - b) / S)
            if w:
                for i in range(2):
                    for j in range(2):
                        ref[i][j] += w * _V[a][i] * _V[b][j]
    dof = T / float(T - 2)
    for i in range(2):
        for j in range(2):
            assert float(got[i][j]) == pytest.approx(dof * ref[i][j] / T,
                                                     abs=1e-9)


def test_a_sub_unit_bandwidth_leaves_only_gamma_zero():
    got = kernel_hac(_V, 0.9, kernel="bartlett", n_params=0)
    T = len(_V)
    for i in range(2):
        for j in range(2):
            want = sum(_V[t][i] * _V[t][j] for t in range(T)) / T
            assert float(got[i][j]) == pytest.approx(want, abs=1e-12)


def test_kernel_hac_validation():
    with pytest.raises(ValueError, match="kernel must be one of"):
        kernel_hac(_V, 2.0, kernel="box")
    with pytest.raises(ValueError, match="bandwidth"):
        kernel_hac(_V, 0.0)
    with pytest.raises(ValueError, match="estimated parameters"):
        kernel_hac(_V, 2.0, n_params=len(_V))
    with pytest.raises(ValueError, match="no observations"):
        kernel_hac([], 2.0)


# --------------------------------------------------------------------------
# alpha(q)
# --------------------------------------------------------------------------

def _ar1_series(rho, n=400, seed=21):
    out = []
    prev = 0.0
    for v in random_normal(n, seed=seed, stream=0):
        prev = rho * prev + float(v)
        out.append([prev])
    return out


def test_alpha_reduces_to_its_closed_form_for_one_series():
    ser = _ar1_series(0.7)
    a2, fits = alpha_ar1(ser, q=2)
    rho, _ = fits[0]
    assert a2 == pytest.approx(4.0 * rho ** 2 / (1.0 - rho) ** 4, rel=1e-12)
    a1, _ = alpha_ar1(ser, q=1)
    assert a1 == pytest.approx(
        4.0 * rho ** 2 / ((1.0 - rho) ** 2 * (1.0 + rho) ** 2), rel=1e-12)
    assert rho == pytest.approx(0.7, abs=0.06)


def test_alpha_weights_can_drop_the_intercept():
    ser = _ar1_series(0.5)
    two = [[1.0, row[0]] for row in ser]        # column 0 is constant
    dropped, _ = alpha_ar1(two, q=2, weights="drop_first")
    alone, _ = alpha_ar1(ser, q=2)
    assert dropped == pytest.approx(alone, rel=1e-12)


def test_alpha_validation():
    ser = _ar1_series(0.3)
    with pytest.raises(ValueError, match="q = 1 or 2"):
        alpha_ar1(ser, q=3)
    with pytest.raises(ValueError):
        alpha_ar1(ser, weights=[0.0])
    with pytest.raises(ValueError, match="weights"):
        alpha_ar1(ser, weights=[1.0, 1.0])
    with pytest.raises(ValueError):
        ar1_fit([1.0, 2.0])


# --------------------------------------------------------------------------
# the whole estimator
# --------------------------------------------------------------------------

def _var1(A, T, seed=77, burn=300):
    u = [float(v) for v in random_normal(2 * (T + burn), seed=seed, stream=0)]
    out = []
    cur = [0.0, 0.0]
    for t in range(T + burn):
        cur = [A[0][0] * cur[0] + A[0][1] * cur[1] + u[2 * t],
               A[1][0] * cur[0] + A[1][1] * cur[1] + u[2 * t + 1]]
        if t >= burn:
            out.append(list(cur))
    return out


def test_on_a_genuine_var1_the_long_run_variance_is_known():
    A = [[0.6, 0.2], [0.0, 0.5]]
    series = _var1(A, 6000)
    inv = np.linalg.inv(np.eye(2) - np.asarray(A, dtype=float))
    true = np.dot(np.asarray(inv), np.asarray(inv).T)

    r = andrews_monahan_hac(series, prewhiten=True, kernel="qs")
    for i in range(2):
        for j in range(2):
            assert float(r["J"][i][j]) == pytest.approx(
                float(true[i][j]), rel=0.15)
    # the filter really did recover the VAR that generated the data
    for i in range(2):
        for j in range(2):
            assert float(r["A"][0][i][j]) == pytest.approx(A[i][j], abs=0.05)
            assert float(r["D"][i][j]) == pytest.approx(float(inv[i][j]),
                                                        abs=0.12)
    # prewhitening leaves the kernel a nearly white series, so the
    # automatic bandwidth collapses -- the mechanism of the paper
    raw = andrews_monahan_hac(series, prewhiten=False, kernel="qs")
    assert r["bandwidth"] < 0.25 * raw["bandwidth"]


def test_prewhiten_false_is_a_zero_matrix_not_a_special_case():
    r = andrews_monahan_hac(_V, prewhiten=False, kernel="qs")
    assert r["A"] == []
    assert r["var_order"] == 0
    for i in range(2):
        for j in range(2):
            assert float(r["J"][i][j]) == pytest.approx(
                float(r["J_star"][i][j]), abs=1e-12)
            assert float(r["D"][i][j]) == pytest.approx(
                1.0 if i == j else 0.0, abs=1e-12)


def test_the_qs_kernel_gives_a_positive_semidefinite_estimate():
    for seed in (1, 2, 3):
        z = [float(v) for v in random_normal(3 * 150, seed=seed, stream=2)]
        W = [[z[3 * t], z[3 * t + 1] + 0.4 * z[3 * t],
              z[3 * t + 2] - 0.3 * z[3 * t + 1]] for t in range(150)]
        for pw in (True, False):
            r = andrews_monahan_hac(W, prewhiten=pw, kernel="qs")
            ev = [float(v) for v in np.linalg.eigvalsh(np.asarray(r["J"]))]
            assert min(ev) >= -1e-9


def test_the_two_front_ends_agree():
    res = [float(v) for v in random_normal(120, seed=9, stream=0)]
    X = [[1.0, i / 120.0, math.sin(i / 7.0)] for i in range(120)]
    mv = moment_vectors(res, X)
    for t in range(120):
        for j in range(3):
            assert mv[t][j] == pytest.approx(X[t][j] * res[t], abs=1e-15)
    a = andrews_monahan_hac(res, X, weights="drop_first")
    b = andrews_monahan_hac(mv, weights="drop_first", n_params=3)
    for i in range(3):
        for j in range(3):
            assert float(a["J"][i][j]) == pytest.approx(
                float(b["J"][i][j]), abs=1e-12)
    assert a["n_params"] == 3


def test_a_fixed_bandwidth_bypasses_the_plug_in():
    r = andrews_monahan_hac(_V, bandwidth=4.0, kernel="bartlett")
    assert r["bandwidth"] == 4.0
    assert r["bandwidth_automatic"] is False
    assert r["alpha"] is None


def test_var_order_two_still_bounds_the_recolouring():
    series = _var1([[0.7, 0.2], [0.1, 0.6]], 500)
    r = andrews_monahan_hac(series, var_order=2, kernel="qs")
    assert len(r["A"]) == 2
    tot = np.asarray(r["A"][0]) + np.asarray(r["A"][1])
    ev = [abs(complex(v))
          for v in np.linalg.eigvals(np.eye(2) - np.asarray(tot))]
    assert min(ev) >= 0.03 - 1e-9


def test_input_validation():
    with pytest.raises(ValueError, match="kernel must be one of"):
        andrews_monahan_hac(_V, kernel="box")
    with pytest.raises(ValueError, match="no observations"):
        andrews_monahan_hac([])
    with pytest.raises(ValueError, match="ragged"):
        moment_vectors([1.0, 2.0], [[1.0], [1.0, 2.0]])
    with pytest.raises(ValueError, match="residuals"):
        moment_vectors([1.0], [[1.0], [2.0]])
    with pytest.raises(ValueError, match="non-negative"):
        prewhiten_var(_V, order=-1)
    with pytest.raises(ValueError, match="cannot fit"):
        prewhiten_var([[1.0, 2.0], [3.0, 4.0]], order=1)
