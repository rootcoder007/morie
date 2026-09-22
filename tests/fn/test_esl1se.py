"""Tests for esl1se.esl_one_se_rule."""

from morie.fn import _array_core as np

from morie.fn.esl1se import esl_one_se_rule


def test_esl1se_basic():
    """Test basic functionality using the docstring example."""
    cv_err = np.asarray([0.9, 0.55, 0.52, 0.50], dtype=float)
    cv_se = np.asarray([0.1, 0.1, 0.1, 0.1], dtype=float)
    result = esl_one_se_rule(cv_err, cv_se)
    assert isinstance(result, dict)
    # Independent recomputation of the rule from the documented formula.
    err = np.asarray([0.9, 0.55, 0.52, 0.50], dtype=float)
    se = np.asarray([0.1, 0.1, 0.1, 0.1], dtype=float)
    i_min = int(np.argmin(err))
    threshold = float(err[i_min] + se[i_min])
    within = np.flatnonzero(err <= threshold)
    chosen = int(within[0])
    assert result["index_min"] == i_min
    assert result["estimate"] == chosen
    assert result["threshold"] == threshold
    assert result["chosen_error"] == float(err[chosen])
    assert result["min_error"] == float(err[i_min])
    assert result["n_within"] == int(within.size)
    assert result["method"].startswith("1-SE rule")


def test_esl1se_edge():
    """Test edge case where minimum is at the last index."""
    cv_err = np.asarray([0.9, 0.8], dtype=float)
    cv_se = np.asarray([0.01, 0.01], dtype=float)
    result = esl_one_se_rule(cv_err, cv_se)
    assert isinstance(result, dict)
    err = np.asarray([0.9, 0.8], dtype=float)
    se = np.asarray([0.01, 0.01], dtype=float)
    i_min = int(np.argmin(err))
    threshold = float(err[i_min] + se[i_min])
    within = np.flatnonzero(err <= threshold)
    chosen = int(within[0])
    assert result["estimate"] == chosen
    assert result["index_min"] == i_min
    assert result["threshold"] == threshold
