"""Tests for mcausm.multi_mediator_causal."""

import numpy as np
import pytest

from morie.fn.mcausm import multi_mediator_causal


def test_mcausm_basic():
    rng = np.random.default_rng(42)
    n = 3000
    x = rng.normal(size=n)
    m1 = 0.7 * x + rng.normal(scale=0.6, size=n)
    m2 = -0.4 * x + rng.normal(scale=0.6, size=n)
    y = 0.2 * x + 1.0 * m1 + 0.5 * m2 + rng.normal(scale=0.6, size=n)
    out = multi_mediator_causal(x, np.c_[m1, m2], y)
    assert out["k"] == 2
    assert out["indirect"] == pytest.approx([0.7, -0.2], abs=0.06)
    assert out["indirect_total"] == pytest.approx(out["indirect"].sum())


def test_mcausm_edge():
    with pytest.raises(ValueError):
        multi_mediator_causal([1.0] * 5, np.zeros((5, 2)), [1.0] * 5)  # too few obs
