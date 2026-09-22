"""Tests for gh_ap_c2.ghosal_packing_num."""

from morie.fn import _array_core as np

from morie.fn.gh_ap_c2 import ghosal_packing_num


def test_gh_ap_c2_basic():
    """Test basic functionality with scalar radius_set."""
    # The function takes a scalar (or array-like) radius_set, eps, dim.
    # Use a scalar for the simplest case.
    radius_set = np.asarray(2.0, dtype=float)
    eps = np.asarray(0.5, dtype=float)
    dim = 2
    result = ghosal_packing_num(radius_set, eps, dim)

    # Check keys match documented formula output
    assert "estimate" in result
    assert "sandwich" in result
    assert "relation_holds" in result

    # Independent computation of the documented formula:
    #   N(eps)   = (3 * radius / eps)^dim
    #   N(eps/2) = (6 * radius / eps)^dim
    #   D        = N(eps)
    r = float(np.asarray(radius_set, dtype=float))
    e = float(np.asarray(eps, dtype=float))
    d = int(dim)
    N_eps = (3.0 * r / e) ** d
    N_half = (6.0 * r / e) ** d
    expected_estimate = N_eps
    expected_sandwich = [N_eps, N_half]

    est = float(np.asarray(result["estimate"], dtype=float))
    assert np.all(np.isfinite(np.asarray(est, dtype=float)))
    assert est == expected_estimate

    sandwich = result["sandwich"]
    assert float(np.asarray(sandwich[0], dtype=float)) == expected_sandwich[0]
    assert float(np.asarray(sandwich[1], dtype=float)) == expected_sandwich[1]

    # N(eps) <= N(eps/2) always holds for positive inputs
    assert bool(result["relation_holds"]) is True


def test_gh_ap_c2_edge():
    """Test edge case with a single-element radius set."""
    # The function does not return a key 'n'; it returns 'estimate'.
    radius_set = np.asarray([42.0], dtype=float)
    eps = np.asarray(0.25, dtype=float)
    dim = 2
    result = ghosal_packing_num(radius_set, eps, dim)

    # Documented keys
    assert "estimate" in result
    assert "sandwich" in result
    assert "relation_holds" in result

    # Independent computation
    r = 42.0
    e = 0.25
    d = 2
    N_eps = (3.0 * r / e) ** d   # (3*42/0.25)^2 = 504^2
    N_half = (6.0 * r / e) ** d  # (6*42/0.25)^2 = 1008^2

    est = float(np.asarray(result["estimate"], dtype=float))
    assert est == N_eps

    sandwich = result["sandwich"]
    assert float(np.asarray(sandwich[0], dtype=float)) == N_eps
    assert float(np.asarray(sandwich[1], dtype=float)) == N_half

    assert bool(result["relation_holds"]) is True
