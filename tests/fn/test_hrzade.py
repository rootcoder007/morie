"""Tests for hrzade.hrz_average_derivative (Horowitz Sec. 2.6)."""

from morie.fn import _array_core as np
from morie.fn.hrzade import hrz_average_derivative


def test_hrzade_scales_linearly_in_beta():
    rng = np.random.default_rng(5)
    x = rng.normal(0.0, 1.0, 200)
    d2 = hrz_average_derivative(x, 2.0 * x)
    d4 = hrz_average_derivative(x, 4.0 * x)
    assert d2["delta"] > 0
    assert abs(d4["delta"] / d2["delta"] - 2.0) < 1e-9
    assert d2["proportional_to_beta"] is True


def test_hrzade_sign_follows_slope():
    rng = np.random.default_rng(6)
    x = rng.normal(0.0, 1.0, 200)
    down = hrz_average_derivative(x, -3.0 * x)
    assert down["delta"] < 0
    assert down["se"] > 0
