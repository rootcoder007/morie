"""Tests for wsmasm.wasserman_mle_asymptotic."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.wsmasm import wasserman_mle_asymptotic
from morie.fn.wsmfis import wasserman_fisher_info

# The Wald z used by the module for a 95 percent interval.
_Z = 1.959963984540054


def _grid(theta):
    """The module's own default support for the exponential model,
    coarser so the quadrature stays quick (error ~1e-5, checked below)."""
    return np.linspace(0.0, 40.0 * theta, 5001)


def test_wsmasm_basic():
    """Exponential model: I(theta) = 1/theta^2, so se = theta/sqrt(n)."""
    n = 100
    theta_hat = 2.0
    g = _grid(theta_hat)
    data = [float(i) for i in range(n)]
    out = wasserman_mle_asymptotic(data, None, theta_hat, x_grid=g)

    assert out["n"] == n
    assert out["estimate"] == theta_hat

    # The plug-in information is exactly what the sibling that computes
    # it returns, and close to the analytic 1/theta^2.
    assert out["information"] == pytest.approx(
        wasserman_fisher_info(None, theta_hat, x_grid=g)["estimate"], rel=1e-12
    )
    assert abs(out["information"] - 1.0 / theta_hat ** 2) < 1e-4

    # se = 1/sqrt(n I), which for this model is theta/sqrt(n).
    assert out["se"] == pytest.approx(
        1.0 / math.sqrt(n * out["information"]), rel=1e-12
    )
    assert abs(out["se"] - theta_hat / math.sqrt(n)) < 1e-4

    # The Wald interval is symmetric about the estimate, half-width z*se.
    assert out["ci_lower"] == pytest.approx(theta_hat - _Z * out["se"], rel=1e-12)
    assert out["ci_upper"] == pytest.approx(theta_hat + _Z * out["se"], rel=1e-12)
    assert out["ci_lower"] < theta_hat < out["ci_upper"]
    assert out["ci_upper"] - out["ci_lower"] == pytest.approx(
        2.0 * _Z * out["se"], rel=1e-12
    )


def test_wsmasm_edge():
    """n = 1 leaves se = 1/sqrt(I); se shrinks as 1/sqrt(n)."""
    theta_hat = 3.0
    g3 = _grid(theta_hat)
    one = wasserman_mle_asymptotic([0.0], None, theta_hat, x_grid=g3)
    assert one["n"] == 1
    assert one["se"] == pytest.approx(
        1.0 / math.sqrt(one["information"]), rel=1e-12
    )
    assert abs(one["se"] - theta_hat / math.sqrt(1)) < 1e-3

    # Quadrupling n halves the standard error at a fixed theta_hat.
    g15 = _grid(1.5)
    small = wasserman_mle_asymptotic(list(range(25)), None, 1.5, x_grid=g15)
    large = wasserman_mle_asymptotic(list(range(100)), None, 1.5, x_grid=g15)
    assert small["se"] / large["se"] == pytest.approx(2.0, rel=1e-12)

    # Only the sample size enters the se, not the sample values.
    shifted = wasserman_mle_asymptotic(
        [1e6 + i for i in range(100)], None, 1.5, x_grid=g15
    )
    assert shifted["se"] == pytest.approx(large["se"], rel=1e-12)

    with pytest.raises(ValueError):
        wasserman_mle_asymptotic([], None, 1.0, x_grid=g15)
    # The exponential default model needs theta > 0.
    with pytest.raises(ValueError):
        wasserman_mle_asymptotic([1.0, 2.0], None, -1.0, x_grid=g15)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.wsmasm as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
