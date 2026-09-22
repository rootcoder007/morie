"""Tests for gh_c11_8.ghosal_fbm_prior."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.gh_c11_8 import ghosal_fbm_prior


def test_gh_c11_8_basic():
    """Test basic functionality."""
    ts = (0.25, 0.5, 0.75)
    result = ghosal_fbm_prior(H=0.7, ts=ts)
    assert "estimate" in result
    estimate = float(np.asarray(result["estimate"], dtype=float))
    # Independent computation of K(ts[0], ts[0]) via the documented formula:
    H = 0.7
    expected_estimate = 0.5 * (ts[0] ** (2 * H) + ts[0] ** (2 * H)
                                - abs(ts[0] - ts[0]) ** (2 * H))
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    assert abs(estimate - expected_estimate) < 1e-12

    # Kernel matrix shape matches the input
    kernel = result["kernel"]
    assert len(kernel) == len(ts)
    for row in kernel:
        assert len(row) == len(ts)

    # variance gap should be ~0 since K(t,t) == t^{2H} by construction
    assert abs(result["var_gap"]) < 1e-12

    # positive definiteness on the grid
    assert result["positive_definite"] is True

    # method key documented in the function
    assert "method" in result


def test_gh_c11_8_edge():
    """Test edge cases."""
    result = ghosal_fbm_prior(H=0.5, ts=(42.0,))
    # estimate is the first diagonal element: K(42, 42) = 0.5*(42^{2H}+42^{2H}-0) = 42^{2H}
    H = 0.5
    expected = 42.0 ** (2 * H)
    estimate = float(np.asarray(result["estimate"], dtype=float))
    assert abs(estimate - expected) < 1e-12
    assert result["positive_definite"] is True
