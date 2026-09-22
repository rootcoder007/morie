"""Tests for gh_c8_14.ghosal_convex_misp."""

from morie.fn import _array_core as np

from morie.fn.gh_c8_14 import ghosal_convex_misp


def test_gh_c8_14_basic():
    """Test basic functionality."""
    # Reference (p0) and two candidate distributions (q1, q2) on the same simplex.
    p0 = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
    q1 = np.array([0.5, 0.2, 0.1, 0.1, 0.1])
    q2 = np.array([0.1, 0.1, 0.1, 0.2, 0.5])
    n_grid = 51

    result = ghosal_convex_misp(p0, q1, q2, n_grid=n_grid)

    # The function must return an "estimate" key.
    assert "estimate" in result

    est = float(np.asarray(result["estimate"], dtype=float))
    # The minimizing mixture lies in [0, 1].
    assert np.all(np.isfinite(np.asarray(est)))
    assert 0.0 <= est <= 1.0

    # Independent recomputation of the KL projection along the segment,
    # using the same grid and formula written out by hand. The minimum of
    # this independently computed curve must agree with the returned
    # estimate to within half a grid spacing.
    ts_indep = [i / (n_grid - 1.0) for i in range(n_grid)]
    kls_indep = []
    for t in ts_indep:
        q = (1.0 - t) * q1 + t * q2
        kl = 0.0
        for x, y in zip(p0, q):
            if x > 0:
                y_safe = y if y > 1e-300 else 1e-300
                kl += x * np.log(x / y_safe) if False else x * np.log(x / y_safe)
        kls_indep.append(kl)
    t_min_indep = ts_indep[kls_indep.index(min(kls_indep))]

    grid_step = 1.0 / (n_grid - 1.0)
    assert abs(est - t_min_indep) <= 0.5 * grid_step + 1e-12

    # The function documents convexity along the segment; include it
    # explicitly as an additional assertion.
    assert "convex_along_segment" in result
    assert bool(result["convex_along_segment"]) is True


def test_gh_c8_14_edge():
    """Test edge cases."""
    # Minimal non-degenerate call: single-element distributions.
    p0 = np.array([1.0])
    q1 = np.array([1.0])
    q2 = np.array([1.0])
    result = ghosal_convex_misp(p0, q1, q2, n_grid=11)
    # The KL divergence is identically zero, so the minimizer is at t=0.
    assert "estimate" in result
    assert float(np.asarray(result["estimate"], dtype=float)) == 0.0
    assert "convex_along_segment" in result
    assert bool(result["convex_along_segment"]) is True
