"""Tests for fgsbh.fine_gray_subdistribution_hazard."""

from morie.fn import _array_core as np
from morie.fn.fgsbh import fine_gray_subdistribution_hazard


def test_fgsbh_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p = 100, 5
    X = rng.normal(0, 1, (n, p))
    time = rng.uniform(0.1, 10, n)
    cause = rng.integers(0, 3, n)
    result = fine_gray_subdistribution_hazard(time, cause, X, of_cause=1)
    assert isinstance(result, dict)
    assert "baseline_cif" in result
    assert "beta" in result
    cif = result["baseline_cif"]
    assert min(cif) >= 0
    assert max(cif) <= 1


def test_fgsbh_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    X = rng.normal(0, 1, (n, p))
    time = np.linspace(0.1, 10, n)
    cause = rng.integers(0, 3, n)
    result = fine_gray_subdistribution_hazard(time, cause, X, of_cause=2)
    assert isinstance(result, dict)
    assert "cumulative_incidence" in result
    assert "times" in result
    assert "se" in result
