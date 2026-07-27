"""Tests for deciA.deci_model."""

import numpy as np
import pytest

from morie.fn.deciA import deci_model


def _confounded(seed=42, n=4000):
    rng = np.random.default_rng(seed)
    conf = rng.normal(size=n)
    t = 0.9 * conf + rng.normal(scale=0.6, size=n)
    y = 1.0 * t + 1.5 * conf + rng.normal(scale=0.5, size=n)
    return np.column_stack([conf, t, y])


def test_deciA_basic():
    out = deci_model(_confounded(), "T", "Y", names=["U", "T", "Y"])
    assert "U" in out["adjustment_set"]
    assert out["estimate"] == pytest.approx(1.0, abs=0.1)
    assert abs(out["naive"] - 1.0) > 0.3


def test_deciA_edge():
    data = _confounded()
    ver = deci_model(data, "T", "Y", names=["U", "T", "Y"], dag={"U": ["T", "Y"], "T": ["Y"]})
    assert ver["backdoor_verified"] is True
    with pytest.raises(ValueError):
        deci_model(data, "T", "T", names=["U", "T", "Y"])
