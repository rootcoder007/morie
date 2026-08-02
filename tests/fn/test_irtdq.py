"""Tests for irtdq."""

from morie.fn import _array_core as np
import pytest

from morie.fn.irtdq import irt_quadratic_utility


def test_irtdq_basic():
    out = irt_quadratic_utility(0.5, yea_position=0.0, nay_position=1.0)
    assert out["p_yea"] == pytest.approx(0.5)
    assert irt_quadratic_utility(-1.0, 0.0, 1.0)["p_yea"] > 0.9  # near yea


def test_irtdq_edge():
    with pytest.raises(ValueError):
        irt_quadratic_utility(0.0, 1.0, 1.0)  # coincident outcomes
    with pytest.raises(ValueError):
        irt_quadratic_utility(0.0, 0.0, 1.0, noise_sd=0.0)
