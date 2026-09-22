"""Tests for fwxF.fire_weather_index."""

from morie.fn import _array_core as np

from morie.fn.fwxF import fire_weather_index


def test_fwxF_basic():
    """Test basic functionality."""
    rng_T = np.random.default_rng(43)
    rng_M = np.random.default_rng(42)
    T = rng_T.integers(15, 25, 100).astype(float)
    RH = rng_M.uniform(30, 70, 100)
    wind = rng_M.uniform(5, 15, 100)
    precip = rng_M.uniform(0, 2, 100)
    month = rng_M.integers(1, 13, 100)
    result = fire_weather_index(T, RH, wind, precip, month)
    assert isinstance(result, dict)
    # The documented return keys are ffmc, dmc, dc, isi, bui, fwi, dsr, n_days
    for key in ("ffmc", "dmc", "dc", "isi", "bui", "fwi", "dsr", "n_days"):
        assert key in result
    assert result["n_days"] == 100
    assert len(result["ffmc"]) == 100
    assert len(result["dmc"]) == 100
    assert len(result["dc"]) == 100
    assert len(result["isi"]) == 100
    assert len(result["bui"]) == 100
    assert len(result["fwi"]) == 100
    assert len(result["dsr"]) == 100
    # Verify DSR = 0.0272 * FWI^1.77 (independent expression)
    fwi_vals = result["fwi"]
    dsr_vals = result["dsr"]
    for fwi, dsr in zip(fwi_vals, dsr_vals):
        expected_dsr = 0.0272 * (fwi ** 1.77)
        assert abs(dsr - expected_dsr) < 1e-9


def test_fwxF_edge():
    """Test edge cases with a single day and scalar month."""
    T = [20.0]
    RH = [50.0]
    wind = [10.0]
    precip = [0.0]
    result = fire_weather_index(T, RH, wind, precip, 7)
    assert isinstance(result, dict)
    assert result["n_days"] == 1
    assert len(result["ffmc"]) == 1
    assert len(result["fwi"]) == 1
    assert len(result["dsr"]) == 1
    for key in ("ffmc", "dmc", "dc", "isi", "bui", "fwi", "dsr", "n_days"):
        assert key in result
    # Independent check on DSR relationship
    expected_dsr = 0.0272 * (result["fwi"][0] ** 1.77)
    assert abs(result["dsr"][0] - expected_dsr) < 1e-9
