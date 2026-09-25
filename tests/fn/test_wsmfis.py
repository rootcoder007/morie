"""Tests for wsmfis.wasserman_fisher_info."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.wsmfis import wasserman_fisher_info


def test_wsmfis_basic():
    """Exponential model: I(theta) = 1/theta^2 exactly."""
    for theta in (0.5, 1.0, 2.0, 4.0):
        g = np.linspace(0.0, 40.0 * theta, 5001)
        out = wasserman_fisher_info(None, theta, x_grid=g)
        assert out["theta"] == theta
        assert out["grid_points"] == 5001
        assert abs(out["estimate"] - 1.0 / theta ** 2) < 1e-4
        # se for a single observation is 1/sqrt(I) = theta here.
        assert out["se_one_obs"] == pytest.approx(
            1.0 / math.sqrt(out["estimate"]), rel=1e-12
        )
        assert abs(out["se_one_obs"] - theta) < 1e-3


def test_wsmfis_custom_normal_density():
    """N(theta, 1) has I(theta) = 1 at every theta."""
    def normal(x, th):
        return np.exp(-0.5 * (x - th) ** 2) / math.sqrt(2.0 * math.pi)

    for theta in (-2.0, 0.0, 3.0):
        g = np.linspace(theta - 10.0, theta + 10.0, 4001)
        out = wasserman_fisher_info(normal, theta, x_grid=g)
        assert out["theta"] == theta
        assert abs(out["estimate"] - 1.0) < 1e-4
        assert abs(out["se_one_obs"] - 1.0) < 1e-4


def test_wsmfis_edge():
    """A custom density without a grid, and a non-positive theta, are errors."""
    with pytest.raises(ValueError):
        wasserman_fisher_info(lambda x, th: None, 1.0)
    with pytest.raises(ValueError):
        wasserman_fisher_info(None, 0.0)
    with pytest.raises(ValueError):
        wasserman_fisher_info(None, -1.0)

    # Halving theta quadruples the information (I = 1/theta^2).
    a = wasserman_fisher_info(None, 2.0, x_grid=np.linspace(0.0, 80.0, 5001))
    b = wasserman_fisher_info(None, 1.0, x_grid=np.linspace(0.0, 40.0, 5001))
    assert b["estimate"] / a["estimate"] == pytest.approx(4.0, rel=1e-3)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.wsmfis as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
