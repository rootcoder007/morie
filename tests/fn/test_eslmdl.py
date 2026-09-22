"""Tests for eslmdl.esl_mdl."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.eslmdl import esl_mdl


def test_eslmdl_basic():
    """Test basic functionality: BIC-equivalent MDL with standard parameter cost."""
    loglik = -120.0
    theta = 4
    n = 100
    result = esl_mdl(loglik, theta, n=n)

    # Expected values computed from the documented formula.
    d = 4
    expected_mdl = -loglik + 0.5 * d * np.log(n)
    expected_bic = d * np.log(n) - 2 * loglik
    expected_data_cost = -loglik
    expected_model_cost = 0.5 * d * np.log(n)
    expected_aic = 2 * d - 2 * loglik

    # Result is dict-like (RichResult) with the documented keys.
    assert "mdl" in result
    assert "bits" in result
    assert "data_cost" in result
    assert "model_cost" in result
    assert "bic" in result
    assert "aic" in result
    assert "d" in result

    assert result["mdl"] == expected_mdl
    assert result["bic"] == expected_bic
    assert result["data_cost"] == expected_data_cost
    assert result["model_cost"] == expected_model_cost
    assert result["aic"] == expected_aic
    assert result["d"] == d

    # ESL Sec 7.8 equivalence: MDL = (1/2) * BIC under the standard cost.
    assert abs(result["mdl"] - result["bic"] / 2) < 1e-12


def test_eslmdl_edge():
    """Test edge cases: more parameters cost more at the same log-likelihood."""
    loglik = -120.0
    n = 100
    mdl_4 = esl_mdl(loglik, 4, n=n)["mdl"]
    mdl_8 = esl_mdl(loglik, 8, n=n)["mdl"]
    assert mdl_8 > mdl_4
    # Independent computation of the expected MDL for d=8.
    expected_mdl_8 = -loglik + 0.5 * 8 * np.log(n)
    assert abs(mdl_8 - expected_mdl_8) < 1e-12

    # No sample size -> BIC is NaN, but MDL is still defined via 2*pi.
    r_no_n = esl_mdl(-10.0, 2)
    assert np.isnan(r_no_n["bic"])
    expected_mdl_no_n = -(-10.0) + 0.5 * 2 * np.log(2 * np.pi)
    assert abs(r_no_n["mdl"] - expected_mdl_no_n) < 1e-12
