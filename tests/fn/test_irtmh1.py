"""Tests for irtmh1.dif_mantel_haenszel."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.irtmh1 import dif_mantel_haenszel


def test_irtmh1_basic():
    """Test basic functionality with valid binary inputs."""
    rng = np.random.default_rng(42)
    n = 40
    J = 5
    # X: n x J matrix of 0/1 item responses
    X = rng.integers(0, 2, size=(n, J)).tolist()
    # group: length-n binary vector
    group = rng.integers(0, 2, size=n).tolist()
    # total_score: length-n matching variable (discrete strata)
    total_score = rng.integers(0, 5, size=n).tolist()

    result = dif_mantel_haenszel(X, group, total_score)

    # The function returns a RichResult object; ensure it has a payload dict
    assert hasattr(result, "payload")
    payload = result.payload
    assert isinstance(payload, dict)

    # Check expected keys
    expected_keys = {
        "estimate",
        "odds_ratio",
        "chisq",
        "p_value",
        "delta",
        "ets_class",
        "flagged",
        "n",
        "method",
    }
    assert expected_keys.issubset(payload.keys())

    # Check basic shapes and types
    assert payload["n"] == n
    assert len(payload["odds_ratio"]) == J
    assert len(payload["chisq"]) == J
    assert len(payload["p_value"]) == J
    assert len(payload["delta"]) == J
    assert len(payload["ets_class"]) == J
    assert len(payload["flagged"]) == J

    # estimate is a float, either finite or NaN
    estimate = payload["estimate"]
    assert isinstance(estimate, float)
    assert math.isfinite(estimate) or math.isnan(estimate)

    # ets_class entries are 'A', 'B' or 'C'
    for cls in payload["ets_class"]:
        assert cls in ("A", "B", "C")

    # flagged entries are 0 or 1
    for flag in payload["flagged"]:
        assert flag in (0, 1)

    # method is a string
    assert isinstance(payload["method"], str)


def test_irtmh1_edge():
    """Test that invalid group coding raises ValueError."""
    rng = np.random.default_rng(42)
    n = 40
    J = 5
    X = rng.integers(0, 2, size=(n, J)).tolist()
    # group contains a value other than 0/1 -> invalid
    group = [2] + [0] * (n - 1)
    total_score = rng.integers(0, 5, size=n).tolist()

    with pytest.raises(ValueError):
        dif_mantel_haenszel(X, group, total_score)
