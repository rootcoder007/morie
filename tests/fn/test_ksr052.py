"""Tests for ksr052 (Kosorok shelf)."""

import numpy as np
import pytest

from morie.fn.ksr052 import kosorok_ch2_kaplan_meier_derivative


def test_ksr052_basic():
    out = kosorok_ch2_kaplan_meier_derivative(lambda u: np.exp(-0.5 * u),
                                              lambda u: np.exp(-0.3 * u),
                                              lambda u: u, lambda u: 1.0, 1.0)
    assert out["boundary_term"] == pytest.approx(np.exp(-0.3))


def test_ksr052_edge():
    with pytest.raises(ValueError):
        kosorok_ch2_kaplan_meier_derivative(lambda u: 1.0, lambda u: 1.0,
                                            lambda u: u, lambda u: 1.0, -1.0)
