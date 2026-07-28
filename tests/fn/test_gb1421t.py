"""Tests for gb1421t (Gibbons shelf)."""

import numpy as np
import pytest

from morie.fn.gb1421t import gibbons_phi_cramers_v


def test_gb1421t_basic():
    out = gibbons_phi_cramers_v([[18, 7], [6, 19]])
    assert out["phi"] == pytest.approx(out["cramers_v"])  # equal on 2x2


def test_gb1421t_edge():
    assert gibbons_phi_cramers_v([[25, 25], [25, 25]])["cramers_v"] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        gibbons_phi_cramers_v([[-1, 2], [3, 4]])
