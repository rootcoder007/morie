"""Tests for svymed.survey_median."""

from morie.fn import _array_core as np
from morie.fn.svymed import survey_median


def test_svymed_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = survey_median(y, weights)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_svymed_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = survey_median(y, weights)
    assert isinstance(result, dict)


def test_weighted_median_and_mean():
    import pytest

    y = [3.0, 1.0, 4.0, 2.0, 5.0]
    w = [1.0, 3.0, 1.0, 1.0, 2.0]
    o = sorted(range(5), key=lambda i: y[i])
    tot, cum, med = sum(w), 0.0, None
    for i in o:
        cum += w[i]
        if cum / tot >= 0.5:
            med = y[i]
            break
    r = survey_median(y, w)
    assert r["estimate"] == med
    assert r["mean"] == pytest.approx(sum(a * b for a, b in zip(w, y)) / tot, rel=1e-14)
