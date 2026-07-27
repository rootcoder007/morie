"""Tests for msmest.marginal_structural_model."""

import numpy as np
import pytest

from morie.fn.msmest import marginal_structural_model


def test_msmest_basic():
    # point treatment confounded by L; IPTW-MSM recovers effect 2
    rng = np.random.default_rng(42)
    L = rng.normal(size=2500)
    A = (rng.random(2500) < 1 / (1 + np.exp(-1.5 * L))).astype(float)
    y = 2.0 * A + 1.5 * L + rng.normal(scale=0.5, size=2500)
    result = marginal_structural_model(y, A, L)
    assert result["estimate"] == pytest.approx(2.0, abs=0.3)  # measured ~2.02
    assert result["weights"].mean() == pytest.approx(1.0, abs=0.15)


def test_msmest_edge():
    with pytest.raises(ValueError):
        marginal_structural_model([1.0, 2.0], [0.5, 1.0], [0.0, 0.0])  # non-binary A
    with pytest.raises(ValueError):
        marginal_structural_model([1.0], [[1, 0]], [[0.0]])  # shape mismatch
