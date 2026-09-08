"""Tests for causivar.causal_iv_anderson_rubin.

Anchored on the just-identified case: with a single instrument the AR
statistic is the square of the t statistic on Z in the regression of
(y - beta0 D) on Z.  That t is built here from scratch.
"""

import pytest

from morie.fn import _array_core as np
from morie.fn.causivar import causal_iv_anderson_rubin

N = 40
_I = list(range(N))
Z0 = [0.2 + 0.09 * i + 0.25 * ((i * i) % 7) for i in _I]
Z1 = [1.1 - 0.05 * i + 0.4 * ((i * i) % 7) for i in _I]
D = [0.8 * Z0[i] + 0.6 * Z1[i] + 0.3 * ((i * 2) % 3) for i in _I]
BETA = 1.5
Y = [BETA * D[i] + 0.4 * ((i * 5) % 4) for i in _I]
Z = np.column_stack([np.asarray(Z0), np.asarray(Z1)])


def test_just_identified_ar_equals_t_squared():
    z = np.asarray(Z0, dtype=float).reshape(N, 1)
    resid = np.asarray(Y, dtype=float) - BETA * np.asarray(D, dtype=float)
    Dz = np.column_stack([np.ones(N), z])
    b, *_ = np.linalg.lstsq(Dz, resid, rcond=None)
    r = resid - Dz @ b
    s2 = float(r @ r) / (N - 2)
    t = float(b[1]) / (s2 * float(np.linalg.inv(Dz.T @ Dz)[1, 1])) ** 0.5

    res = causal_iv_anderson_rubin(Y, D, z, BETA)
    assert res["statistic"] == pytest.approx(t * t, rel=1e-9)


def test_ar_does_not_reject_at_the_true_beta():
    res = causal_iv_anderson_rubin(Y, D, Z, BETA)
    assert res["p_value"] > 0.05


def test_ar_rejects_far_from_the_true_beta():
    res = causal_iv_anderson_rubin(Y, D, Z, 0.0)
    assert res["p_value"] < 0.01
    assert res["statistic"] > causal_iv_anderson_rubin(Y, D, Z, BETA)["statistic"]


def test_ar_needs_degrees_of_freedom():
    with pytest.raises(ValueError):
        causal_iv_anderson_rubin(Y[:3], D[:3], Z[:3], BETA)
