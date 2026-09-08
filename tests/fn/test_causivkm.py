"""Tests for causivkm.causal_iv_kleibergen_moreira (Moreira 2003 CLR).

Anchored on Moreira (2003): in the just-identified case (one
instrument) the conditional LR test IS the Anderson-Rubin test, so the
two p-values must coincide exactly.  That is a property of the method,
not of this implementation, and it exercises the conditional p-value
branch as well as the statistic.
"""

import pytest

from morie.fn import _array_core as np
from morie.fn.causivar import causal_iv_anderson_rubin
from morie.fn.causivkm import causal_iv_kleibergen_moreira

N = 40
_I = list(range(N))
Z0 = [0.2 + 0.09 * i + 0.25 * ((i * i) % 7) for i in _I]
Z1 = [1.1 - 0.05 * i + 0.4 * ((i * i) % 7) for i in _I]
Z2 = [0.3 + 0.02 * i + 0.5 * ((i * i * i) % 11) for i in _I]
D = [0.8 * Z0[i] + 0.6 * Z1[i] + 0.2 * Z2[i] + 0.3 * ((i * 2) % 3) for i in _I]
BETA = 1.5
Y = [BETA * D[i] + 0.4 * ((i * 5) % 4) for i in _I]


def _Z(k):
    return np.column_stack([np.asarray(c) for c in ([Z0], [Z0, Z1], [Z0, Z1, Z2])[k - 1]])


def test_just_identified_clr_is_the_ar_test():
    """Moreira (2003): L = 1 collapses the CLR test onto AR."""
    z = _Z(1)
    clr = causal_iv_kleibergen_moreira(Y, D, z, BETA)
    ar = causal_iv_anderson_rubin(Y, D, z, BETA)
    assert clr["p_value"] == pytest.approx(ar["p_value"], abs=1e-12)


def test_conditional_pvalue_is_a_probability_for_l2_and_l3():
    for k in (2, 3):
        res = causal_iv_kleibergen_moreira(Y, D, _Z(k), BETA)
        assert 0.0 <= res["p_value"] <= 1.0
        assert res["n_instruments"] == k


def test_clr_does_not_reject_at_the_true_beta_but_does_far_away():
    z = _Z(3)
    assert causal_iv_kleibergen_moreira(Y, D, z, BETA)["p_value"] > 0.05
    assert causal_iv_kleibergen_moreira(Y, D, z, 0.0)["p_value"] < 0.01


def test_clr_statistic_is_nonnegative():
    for b in (-2.0, 0.0, BETA, 4.0):
        assert causal_iv_kleibergen_moreira(Y, D, _Z(2), b)["statistic"] >= 0.0
