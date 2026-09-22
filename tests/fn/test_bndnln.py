"""Tests for bndnln.bound_nonlinear."""

import numpy as np

from morie.fn.bndnln import bound_nonlinear


def _moment_fn(data, theta):
    """A trivial moment function with J=2 inequalities.

    Returns an (n, 2) array whose first column is the constant moment
    `theta - 1` (negative iff theta <= 1) and whose second column is
    `theta - 2` (negative iff theta <= 2). Both are stacked per-row so
    that `bound_nonlinear` sees the correct shape regardless of the
    orientation of `g`.
    """
    n = data.shape[0]
    return np.column_stack([
        np.full(n, theta - 1.0),
        np.full(n, theta - 2.0),
    ])


def test_bndnln_basic():
    """Test basic functionality.

    A point well inside the identified set (theta=0, so both population
    moments are negative) must give Q_n = 0 exactly and lie inside the
    confidence region. A point well outside (theta=3) must give Q_n > 0.
    """
    rng_x = np.random.default_rng(42)
    data = rng_x.normal(0.0, 1.0, (200, 3))

    result = bound_nonlinear(data, _moment_fn, theta_grid=0.0)

    # Documented keys.
    for key in ("theta_grid", "criterion", "critical_value",
                "in_confidence_set", "set_estimate",
                "confidence_set_bounds", "n_binding_max", "n", "J",
                "method"):
        assert key in result, f"missing key {key!r} in result"

    # Deep inside the identified set: every sample moment equals its
    # population value (theta-1, theta-2) = (-1, -2) < 0, so t<0,
    # the positive part is zero and Q_n is exactly zero.
    Q_at_zero = float(result["criterion"][0])
    assert Q_at_zero == 0.0, f"expected Q_n(theta=0)=0, got {Q_at_zero}"

    # Confidence region covers the criterion's argmin (the set estimate).
    argmin = result["set_estimate"]
    in_set = result["in_confidence_set"]
    assert bool(in_set[0]) is True, "theta=0 must be in confidence set"
    assert 0.0 in np.atleast_1d(argmin), "set_estimate must include theta=0"

    # Independent recomputation: with data irrelevant (moments are
    # constant across rows) and theta=0, each column is identically -1
    # and -2, so the t-statistics are both 0/1 = 0 and the positive-part
    # sum is 0.
    expected_Q = float((max(-1.0, 0.0) ** 2) + (max(-2.0, 0.0) ** 2))
    assert Q_at_zero == expected_Q

    # A point outside the identified set must give a strictly positive
    # criterion, computed independently: moments are (3-1, 3-2) = (2, 1),
    # SD = 0 each, so the function substitutes 1.0 and t = sqrt(n)*2 and
    # sqrt(n)*1, giving the independent value below.
    result_outside = bound_nonlinear(
        data, _moment_fn, theta_grid=3.0, B=50, seed=0,
    )
    Q_outside = float(result_outside["criterion"][0])
    n = 200
    expected_Q_outside = (np.sqrt(n) * 2.0) ** 2 + (np.sqrt(n) * 1.0) ** 2
    assert Q_outside == expected_Q_outside
    assert Q_outside > 0.0


def test_bndnln_edge():
    """Test edge cases: small alpha boundary, grid vector shape, and
    that a vector `theta_grid` is preserved on the returned object."""
    rng_x = np.random.default_rng(42)
    data = rng_x.normal(0.0, 1.0, (100, 3))

    grid = np.linspace(-0.5, 0.5, 5)
    result = bound_nonlinear(data, _moment_fn, theta_grid=grid,
                             alpha=0.1, B=50, seed=1)

    # Confidence-set bounds should bracket the grid values that are
    # inside the region; for this dataset every grid point sits inside
    # the identified set, so the bounds must cover the whole grid.
    lo, hi = result["confidence_set_bounds"]
    assert lo <= float(grid.min())
    assert hi >= float(grid.max())

    # n and J should reflect what we passed in.
    assert int(result["n"]) == 100
    assert int(result["J"]) == 2

    # In all-grid case every Q_n(theta) is exactly zero.
    assert np.all(np.asarray(result["criterion"]) == 0.0)
