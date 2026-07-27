"""Tests for potef.potential_outcomes_individual."""

import numpy as np
import pytest

from morie.fn.potef import potential_outcomes_individual


def test_potef_basic():
    Y0 = np.array([1.0, 2.0, 3.0])
    Y1 = np.array([2.0, 4.0, 6.0])
    out = potential_outcomes_individual(Y1, Y0)
    assert out["ite"] == pytest.approx([1.0, 2.0, 3.0])
    assert out["ate"] == pytest.approx(2.0)


def test_potef_edge():
    with pytest.raises(ValueError):
        potential_outcomes_individual([1.0, 2.0], [1.0])  # length mismatch
    with pytest.raises(ValueError):
        potential_outcomes_individual([1.0, 2.0], [0.0, 0.0], observed_treatment=[1, 1])
