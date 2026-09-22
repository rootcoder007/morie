"""Tests for morie.fn.splrg — Spline regression."""

from morie.fn import _array_core as np
import pytest

from morie.fn.splrg import splrg


def test_returns_dict():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 100)
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.1, 100)

    # Provide explicit knots to avoid relying on np.percentile
    # (which in this numpy-like shim only accepts scalar arguments).
    knots = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    result = splrg(x, y, knots=knots)
    assert isinstance(result, dict)
    for key in ("x_eval", "y_hat", "coefficients", "knots", "penalty", "n_obs"):
        assert key in result

    # Documented return-shape checks.
    assert result["n_obs"] == 100
    assert len(result["knots"]) == 5
    assert result["penalty"] == 0.0
    # x_eval defaults to sorted x, so length must equal n_obs.
    assert len(result["x_eval"]) == 100
    assert len(result["y_hat"]) == 100
    # coefficients must contain (1 intercept + 1 slope + K truncated-power terms).
    assert len(result["coefficients"]) == 2 + 5


def test_smoothing_spline():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 100)
    y = x ** 2 + rng.normal(0, 0.3, 100)
    # Use explicit knots so we don't trigger the shim's scalar-percentile limit.
    knots = np.array([0.2, 0.4, 0.6, 0.8])
    result = splrg(x, y, knots=knots, penalty=10.0)
    assert float(result["penalty"]) == 10.0

    # Sanity: y_hat should be finite for a solvable penalized system.
    y_hat = np.asarray(result["y_hat"], dtype=float)
    assert np.all(np.isfinite(y_hat))


def test_custom_knots():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 50)
    y = x + rng.normal(0, 0.1, 50)
    knots = np.array([0.25, 0.5, 0.75])
    result = splrg(x, y, knots=knots)
    assert len(result["knots"]) == 3
    # The returned knots must be the ones we passed in.
    assert np.allclose(np.asarray(result["knots"]), knots)


def test_too_few_raises():
    with pytest.raises(ValueError, match="at least 4"):
        splrg(np.ones(3), np.ones(3))
