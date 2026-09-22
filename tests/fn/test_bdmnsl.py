"""Tests for bdmnsl.bound_monot_selection."""

from morie.fn import _array_core as np

from morie.fn.bdmnsl import bound_monot_selection


def test_bdmnsl_basic():
    """Test basic functionality against the documented Manski-Pepper formula."""
    rng_y = np.random.default_rng(43)
    rng_z = np.random.default_rng(42)
    y = rng_y.normal(0, 1, 100)
    z = rng_z.integers(0, 3, 100)  # treatment levels in {0, 1, 2}
    d = 1.0
    ymin = -3.0
    ymax = 3.0

    result = bound_monot_selection(y, z, d, ymin, ymax)

    # Documented return keys
    for key in ("lower", "upper", "width", "condmean",
                "pbelow", "pat", "pabove", "n", "d"):
        assert key in result

    # Independent recomputation of the documented formula
    at_idx = [i for i in range(len(y)) if z[i] == d]
    condmean = sum(y[i] for i in at_idx) / len(at_idx)
    pbelow = sum(1 for i in range(len(y)) if z[i] < d) / len(y)
    pabove = sum(1 for i in range(len(y)) if z[i] > d) / len(y)
    pat = len(at_idx) / len(y)
    expected_upper = (pbelow + pat) * condmean + pabove * ymax
    expected_lower = pbelow * ymin + (pat + pabove) * condmean

    assert result["lower"] == expected_lower
    assert result["upper"] == expected_upper
    assert result["width"] == expected_upper - expected_lower
    assert result["condmean"] == condmean
    assert result["pbelow"] == pbelow
    assert result["pabove"] == pabove
    assert result["pat"] == pat
    assert result["n"] == 100
    assert result["d"] == d


def test_bdmnsl_edge():
    """Test edge case: d is realised and bounds are well-defined."""
    y = np.array([0.1, 0.2, -0.3, 0.4, 0.5])
    z = np.array([0, 1, 1, 2, 2])
    d = 1
    ymin = -1.0
    ymax = 1.0

    result = bound_monot_selection(y, z, d, ymin, ymax)

    assert isinstance(result, dict)
    at_idx = [i for i in range(len(y)) if z[i] == d]
    condmean = sum(y[i] for i in at_idx) / len(at_idx)
    pbelow = sum(1 for i in range(len(y)) if z[i] < d) / len(y)
    pabove = sum(1 for i in range(len(y)) if z[i] > d) / len(y)
    pat = len(at_idx) / len(y)
    expected_upper = (pbelow + pat) * condmean + pabove * ymax
    expected_lower = pbelow * ymin + (pat + pabove) * condmean

    assert result["lower"] == expected_lower
    assert result["upper"] == expected_upper
    assert result["width"] == expected_upper - expected_lower
