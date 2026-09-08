"""Tests for htbias.hettest_bias (BLP calibration test).

Anchored on the exact algebraic identity the grf construction rests on:

    (W - What) * tau
        = (W - What) * mean(tau) + (W - What) * (tau - mean(tau))

so if the residualised outcome IS (W - What) * tau, both regression
coefficients must come out at exactly 1.  And on HC3 itself, rebuilt
from MacKinnon and White (1985).
"""

import pytest

from morie.fn import _array_core as np
from morie.fn.htbias import hettest_bias

N = 60
_I = list(range(N))
D = [float(i % 2) for i in _I]
TAU = [0.5 + 0.04 * i for i in _I]
Y = [TAU[i] * D[i] + 0.3 * ((i * 7) % 5) for i in _I]


def test_exact_grf_identity_gives_unit_coefficients():
    d = np.asarray(D, dtype=float)
    tau = np.asarray(TAU, dtype=float)
    wbar = float(np.mean(d))
    y = (d - wbar) * tau
    res = hettest_bias(y, d, tau, y_hat=np.zeros(N), w_hat=np.full(N, wbar))
    assert res["coef_mean"] == pytest.approx(1.0, abs=1e-10)
    assert res["coef_differential"] == pytest.approx(1.0, abs=1e-10)


def test_hc3_matches_mackinnon_white():
    d = np.asarray(D, dtype=float)
    tau = np.asarray(TAU, dtype=float)
    X = np.column_stack(
        [
            (d - np.mean(d)) * float(np.mean(tau)),
            (d - np.mean(d)) * (tau - float(np.mean(tau))),
        ]
    )
    target = np.asarray(Y, dtype=float) - float(np.mean(Y))
    beta, *_ = np.linalg.lstsq(X, target, rcond=None)
    e = target - X @ beta
    XtXi = np.linalg.inv(X.T @ X)
    H = X @ XtXi @ X.T
    meat = np.zeros((2, 2))
    for i in range(N):
        w = float(e[i]) ** 2 / (1.0 - float(H[i, i])) ** 2
        meat = meat + w * np.outer(X[i, :], X[i, :])
    V = XtXi @ meat @ XtXi

    res = hettest_bias(Y, D, TAU)
    assert res["se_differential"] == pytest.approx(abs(float(V[1, 1])) ** 0.5, abs=1e-10)
    assert res["se_mean"] == pytest.approx(abs(float(V[0, 0])) ** 0.5, abs=1e-10)


def test_p_values_are_one_sided():
    res = hettest_bias(Y, D, TAU)
    # grf converts to one-sided: a positive t gives p = two_sided/2
    assert 0.0 <= res["p_differential"] <= 1.0
    assert res["t_differential"] > 0
    assert res["p_differential"] < 0.5


def test_constant_cate_is_rejected_not_silently_singular():
    with pytest.raises(ValueError):
        hettest_bias(Y, D, [1.0] * N)


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError):
        hettest_bias(Y, D[:10], TAU)


def test_alias_is_the_same_function():
    from morie.fn.htbias import hettestbias

    assert hettestbias is hettest_bias
