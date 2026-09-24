"""Tests for drctf.dr_continuous_treatment."""

from morie.fn import _array_core as np

from morie.fn.drctf import dr_continuous_treatment


def test_drctf_basic():
    """Test basic functionality."""
    rng_y = np.random.default_rng(43)
    rng_d = np.random.default_rng(42)
    rng_x = np.random.default_rng(41)
    n = 100
    y = rng_y.normal(0, 1, n)
    D_dose = rng_d.integers(0, 4, n)  # non-negative doses, zero marks untreated
    X = rng_x.normal(0, 1, (n, 3))
    result = dr_continuous_treatment(y, D_dose, X)
    assert isinstance(result, dict)
    for key in ("estimate", "doses", "att", "se", "acrt", "acrt_dose", "n_zero", "n", "method"):
        assert key in result
    assert isinstance(result["estimate"], float)
    assert result["n"] == n
    assert isinstance(result["doses"], list)
    assert len(result["att"]) == len(result["doses"])
    # acrt length is one less than doses (if there are at least two doses)
    assert len(result["acrt"]) == max(0, len(result["doses"]) - 1)
    assert result["n_zero"] >= 0.0


def test_drctf_edge():
    """Test edge cases."""
    rng_y = np.random.default_rng(43)
    rng_d = np.random.default_rng(42)
    n = 40
    y = rng_y.normal(0, 1, n)
    D_dose = rng_d.integers(0, 3, n)  # non-negative doses, zero marks untreated
    # X omitted (optional)
    result = dr_continuous_treatment(y, D_dose)
    assert isinstance(result, dict)
    for key in ("estimate", "doses", "att", "se", "acrt", "acrt_dose", "n_zero", "n", "method"):
        assert key in result
    assert result["n"] == n
    assert isinstance(result["estimate"], float)
