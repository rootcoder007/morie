"""Tests for gh_c11_13.ghosal_gp_adapt_thm."""

from morie.fn import _array_core as np

from morie.fn.gh_c11_13 import ghosal_gp_adapt_thm


def test_gh_c11_13_basic():
    """Test basic functionality.

    The function `ghosal_gp_adapt_thm` has the signature
    (n=60, l_true=0.2, l_grid=(0.05, 0.2, 0.8), noise=0.1, seed=42).
    It returns a result with at least the key 'estimate' (the MAP
    length scale), 'log_evidence' (one log-evidence per grid point),
    and 'method' (description string).
    """
    result = ghosal_gp_adapt_thm(n=60, l_true=0.2,
                                 l_grid=(0.05, 0.2, 0.8),
                                 noise=0.1, seed=42)

    # The documented return keys must all be present.
    assert "estimate" in result
    assert "log_evidence" in result
    assert "method" in result

    est = float(np.asarray(result["estimate"], dtype=float))
    assert np.isfinite(est)

    # 'estimate' must be one of the values supplied in l_grid.
    assert est in (0.05, 0.2, 0.8)

    # The evidence-weighted MAP should concentrate near the true scale
    # (l_true = 0.2). With seed=42, deterministic evaluation selects
    # the grid point closest to l_true that maximises log evidence.
    assert est == 0.2

    # 'log_evidence' has one entry per grid point, in input order.
    log_ev = [float(np.asarray(v, dtype=float)) for v in result["log_evidence"]]
    assert len(log_ev) == len((0.05, 0.2, 0.8))

    # The MAP estimate must coincide with the grid point that
    # maximises the returned log-evidence values (independent check,
    # computed in the test from the same returned list).
    grid = (0.05, 0.2, 0.8)
    best_index = max(range(len(log_ev)), key=lambda i: log_ev[i])
    assert est == grid[best_index]


def test_gh_c11_13_edge():
    """Test that scalar defaults / repeated calls behave as documented.

    The function takes scalar arguments (n, l_true, noise, seed) and a
    tuple of candidate length scales (l_grid). It does not accept an
    array of observations, and it does not return a key 'n'.
    """
    result = ghosal_gp_adapt_thm()

    # Required keys from the docstring.
    assert "estimate" in result
    assert "log_evidence" in result
    assert "method" in result

    # The estimate must be a scalar drawn from the default grid.
    est = float(np.asarray(result["estimate"], dtype=float))
    assert est in (0.05, 0.2, 0.8)
