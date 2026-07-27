"""Tests for causscg.causal_generalised_sc."""

import numpy as np
import pytest

from morie.fn.causscg import causal_generalised_sc
from morie.fn.gscmcl import generalized_synthetic_control


def test_causscg_basic():
    rng = np.random.default_rng(42)
    Y0 = np.cumsum(rng.normal(size=(30, 6)), axis=0)
    y1 = Y0[:, 0] + rng.normal(scale=0.1, size=30)
    a = causal_generalised_sc(y1, Y0, 20, r=2)
    b = generalized_synthetic_control(y1, Y0, 20, r=2)
    assert a["att"] == pytest.approx(b["att"])


def test_causscg_edge():
    with pytest.raises(ValueError):
        causal_generalised_sc(np.ones(10), np.ones((9, 4)), 5)  # T mismatch
