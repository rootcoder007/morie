"""Tests for gh_c10_13.ghosal_pt_null_tst."""

import math

from morie.fn import _array_core as np

from morie.fn.gh_c10_13 import ghosal_pt_null_tst


def test_gh_c10_13_basic():
    """Test basic functionality with documented scalar inputs."""
    ln = -10.0  # log prod_i p*(X_i)
    la = -15.0  # log int prod_i p(X_i) dPi_1(p)
    lam = 0.5
    result = ghosal_pt_null_tst(ln, la, lam=lam)
    assert isinstance(result, dict)
    # Documented return keys
    assert "log_bayes_factor" in result
    assert "bayes_factor" in result
    assert "posterior_null" in result
    assert "posterior_alt" in result
    assert "prior_null" in result
    assert "lam" in result

    # Independently recompute expected values from the documented formula
    expected_lbf = ln - la
    a = math.log(1.0 - lam) + ln
    b = math.log(lam) + la
    mx = max(a, b)
    denom = mx + math.log(math.exp(a - mx) + math.exp(b - mx))
    expected_post0 = math.exp(a - denom)
    expected_bf = math.exp(expected_lbf)
    expected_prior_null = 1.0 - lam

    assert result["log_bayes_factor"] == expected_lbf
    assert result["bayes_factor"] == expected_bf
    assert result["posterior_null"] == expected_post0
    assert result["posterior_alt"] == 1.0 - expected_post0
    assert result["prior_null"] == expected_prior_null
    assert result["lam"] == lam


def test_gh_c10_13_edge():
    """Test edge cases: lam near boundaries, and large log bayes factor overflow."""
    ln = 0.0
    la = -1000.0  # very small marginal under alternative -> large BF
    lam = 0.9
    result = ghosal_pt_null_tst(ln, la, lam=lam)
    assert isinstance(result, dict)

    expected_lbf = ln - la
    a = math.log(1.0 - lam) + ln
    b = math.log(lam) + la
    mx = max(a, b)
    denom = mx + math.log(math.exp(a - mx) + math.exp(b - mx))
    expected_post0 = math.exp(a - denom)
    expected_prior_null = 1.0 - lam

    assert result["log_bayes_factor"] == expected_lbf
    assert result["bayes_factor"] == math.inf
    assert result["posterior_null"] == expected_post0
    assert result["posterior_alt"] == 1.0 - expected_post0
    assert result["prior_null"] == expected_prior_null
    assert result["lam"] == lam
