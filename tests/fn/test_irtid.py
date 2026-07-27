"""Tests for irtid."""

import numpy as np
import pytest

from morie.fn.irtid import irt_identification_constraints


def test_irtid_basic():
    out = irt_identification_constraints([3.0, 5.0, 7.0, 9.0], polarity_idx=3)
    assert out["x"].mean() == pytest.approx(0.0, abs=1e-12)
    assert out["x"][3] < 0


def test_irtid_edge():
    with pytest.raises(ValueError):
        irt_identification_constraints([1.0, 1.0, 1.0])  # constant
    with pytest.raises(ValueError):
        irt_identification_constraints([1.0, 2.0], polarity_idx=5)  # out of range
