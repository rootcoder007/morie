"""Tests for jntlO.joint_loss_mixed_outcomes."""

import math

from morie.fn import _array_core as np
from morie.fn.jntlO import joint_loss_mixed_outcomes


def test_jntlO_basic():
    """Test basic functionality with all three outcome kinds and explicit weights."""
    rng = np.random.default_rng(42)
    n = 40

    y_cont = rng.normal(0.0, 1.0, n)
    y_bin = rng.integers(0, 2, n)
    y_count = rng.integers(0, 10, n)

    y_dict = {
        "cont": ("cont", y_cont),
        "binary": ("binary", y_bin),
        "count": ("count", y_count),
    }

    y_hat_dict = {
        "cont": rng.normal(0.0, 1.0, n),
        "binary": rng.uniform(0.0, 1.0, n),
        "count": rng.normal(5.0, 2.0, n),
    }

    weights = {"cont": 0.5, "binary": 0.3, "count": 0.2}

    result = joint_loss_mixed_outcomes(y_dict, y_hat_dict, weights)

    assert isinstance(result, dict)
    for key in ("estimate", "loss", "parts", "weights", "n", "method"):
        assert key in result
    assert math.isfinite(result["estimate"])
    assert result["loss"] == result["estimate"]
    assert result["n"] == 3
    assert set(result["parts"].keys()) == {"cont", "binary", "count"}
    assert set(result["weights"].keys()) == {"cont", "binary", "count"}
    assert all(math.isfinite(v) for v in result["parts"].values())


def test_jntlO_edge():
    """Test edge case: omitting weights falls back to the Section 12.4 recipe."""
    rng = np.random.default_rng(7)
    n = 40

    y_dict = {
        "cont": ("cont", rng.normal(0.0, 1.0, n)),
        "binary": ("binary", rng.integers(0, 2, n)),
    }

    y_hat_dict = {
        "cont": rng.normal(0.0, 1.0, n),
        "binary": rng.uniform(0.0, 1.0, n),
    }

    result = joint_loss_mixed_outcomes(y_dict, y_hat_dict)

    assert isinstance(result, dict)
    assert math.isfinite(result["estimate"])
    assert result["n"] == 2
    assert set(result["parts"].keys()) == {"cont", "binary"}
    assert set(result["weights"].keys()) == {"cont", "binary"}
    assert result["weights"]["cont"] == result["weights"]["cont"]
    assert all(w >= 0.0 for w in result["weights"].values())
