"""Tests for bayes_general.bayes_general."""

import math

from morie.fn import _array_core as np

from morie.fn.bayes_general import (
    bayes_general,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e74_basic():
    """Test basic functionality."""
    priors = [0.3, 0.4, 0.3]
    likelihoods = [0.8, 0.5, 0.2]
    result = bayes_general(priors, likelihoods)
    payload = result.payload
    assert isinstance(payload, dict)
    assert "posteriors" in payload
    assert "p_z" in payload
    assert len(payload["posteriors"]) == 3
    assert all(isinstance(p, float) for p in payload["posteriors"])
    assert math.isfinite(payload["p_z"])
    assert 0.0 <= payload["p_z"] <= 1.0
    assert abs(sum(payload["posteriors"]) - 1.0) < 1e-9
    assert all(0.0 <= p <= 1.0 for p in payload["posteriors"])


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e74_edge():
    """Test edge cases."""
    priors = [1.0, 0.0]
    likelihoods = [0.5, 0.5]
    result = bayes_general(priors, likelihoods)
    payload = result.payload
    assert isinstance(payload, dict)
    assert "posteriors" in payload
    assert "p_z" in payload
    assert len(payload["posteriors"]) == 2
    assert math.isfinite(payload["p_z"])
    assert 0.0 <= payload["p_z"] <= 1.0
    assert abs(sum(payload["posteriors"]) - 1.0) < 1e-9
