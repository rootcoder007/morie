"""Tests for hierF. Full anchor: ledger/wave3/anchor_ts_family.py."""
import pytest
from morie.fn import _array_core as np
from morie.fn.hierF import (is_coherent, mint_P, mint_reconcile,
                            shrink_covariance, summing_matrix)


@pytest.fixture(scope="module")
def h():
    rng = np.random.default_rng(11)
    S = summing_matrix([[0, 1, 2], [0, 1], [2]], 3)
    resid = [[rng.standard_normal() * (1.0 + 0.5 * i) for i in range(6)]
             for _ in range(60)]
    return {"S": S, "resid": resid,
            "base": [10.0, 6.5, 4.0, 3.0, 3.0, 3.5]}


def test_the_base_forecasts_are_incoherent(h):
    assert not is_coherent(h["base"], h["S"])


@pytest.mark.parametrize("meth", ["ols", "wls", "shrink"])
def test_ps_is_the_identity_and_the_result_coheres(h, meth):
    r = mint_reconcile(h["base"], h["S"], method=meth,
                       residuals=h["resid"])
    assert r["ps_identity_error"] < 1e-9
    assert r["coherent"]


def test_sp_is_a_projection(h):
    """A wrong P still adds up -- S forces that -- and is simply the
    wrong point in the coherent subspace. Idempotence is what pins it."""
    P, _ = mint_P(h["S"], method="shrink", residuals=h["resid"])
    S = h["S"]
    b = [2.0, 3.0, 4.0]
    coh = [sum(S[a][j] * b[j] for j in range(3)) for a in range(6)]
    r = mint_reconcile(coh, S, method="shrink", residuals=h["resid"])
    for a in range(6):
        assert r["reconciled"][a] == pytest.approx(coh[a], abs=1e-8)
    SP = [[sum(S[a][i] * P[i][c] for i in range(3)) for c in range(6)]
          for a in range(6)]
    SPSP = [[sum(SP[a][c] * SP[c][b2] for c in range(6))
             for b2 in range(6)] for a in range(6)]
    for a in range(6):
        for b2 in range(6):
            assert SPSP[a][b2] == pytest.approx(SP[a][b2], abs=1e-8)


def test_the_weight_matrix_matters(h):
    a = mint_reconcile(h["base"], h["S"], "shrink", h["resid"])
    b = mint_reconcile(h["base"], h["S"], "ols")
    assert max(abs(a["reconciled"][i] - b["reconciled"][i])
               for i in range(6)) > 1e-6
    _, lam = shrink_covariance(h["resid"])
    assert 0.0 <= lam <= 1.0


def test_argument_checks(h):
    with pytest.raises(ValueError):
        mint_reconcile(h["base"], h["S"], method="wls")
    with pytest.raises(ValueError):
        mint_reconcile(h["base"], h["S"], method="nope")
    with pytest.raises(ValueError):
        mint_reconcile(h["base"][:-1], h["S"], "ols")
    with pytest.raises(ValueError):
        summing_matrix([[0, 9]], 3)
