"""Tests for gh_c13_15.ghosal_cox_bvm."""

from morie.fn import _array_core as np

from morie.fn.gh_c13_15 import ghosal_cox_bvm


def test_gh_c13_15_basic():
    """Test basic functionality."""
    time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    event = np.array([1.0, 1.0, 0.0, 1.0, 1.0])
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_cox_bvm(x, time=time, event=event)
    beta = np.asarray(result["beta"], dtype=float)
    se = np.asarray(result["se"], dtype=float)
    bg = np.asarray(result["beta_grid"], dtype=float)
    post = np.asarray(result["posterior_normal"], dtype=float)
    info = np.asarray(result["efficient_information"], dtype=float)

    # Documented return keys must all be present and finite where numeric.
    assert "beta" in result
    assert "se" in result
    assert "efficient_information" in result
    assert "beta_grid" in result
    assert "posterior_normal" in result
    assert result["efficient"] is True
    assert result["credible_equals_confidence"] is True
    assert result["n"] == 5
    assert result["n_events"] == 4

    assert np.all(np.isfinite(beta))
    assert np.all(np.isfinite(se))
    assert np.all(np.isfinite(post))
    assert beta.shape == (1,)
    assert se.shape == (1,)
    assert bg.shape == post.shape

    # Standard normal density on the documented grid, independent of the
    # function's internal evaluation: the function evaluates
    # N(b[0], se[0]^2) on bg, so the log-density minus the log-normalising
    # constant must equal -0.5 * ((bg - beta[0]) / se[0])**2.
    expected_exp = -0.5 * ((bg - beta[0]) / max(se[0], 1e-12)) ** 2
    # log(post * se[0] * sqrt(2 pi)) = -0.5 z^2
    log_unnorm = np.log(post * max(se[0], 1e-12) * np.sqrt(2 * np.pi))
    assert np.all(np.isfinite(log_unnorm))
    assert np.max(np.abs(log_unnorm - expected_exp)) < 1e-8


def test_gh_c13_15_edge():
    """Test that too few observations raises and shape handling works."""
    import pytest
    # Below the documented minimum of 5 observations: must raise.
    with pytest.raises(ValueError):
        ghosal_cox_bvm(np.array([42.0]), time=np.array([1.0]))

    # 1-D covariates become a single-row design matrix; supply matching
    # follow-up times so the function has one observation per time.
    time = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    event = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
    x = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    result = ghosal_cox_bvm(x, time=time, event=event)
    beta = np.asarray(result["beta"], dtype=float)
    assert result["n"] == 8
    assert result["n_events"] == 6
    assert np.all(np.isfinite(beta))
