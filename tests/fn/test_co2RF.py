"""Tests for co2RF.radiative_forcing_co2."""

import math

from morie.fn.co2RF import radiative_forcing_co2


def test_co2RF_basic():
    """Test basic functionality (AR6 method, default reference values)."""
    C = 556.6  # ~2x pre-industrial CO2
    result = radiative_forcing_co2(C)

    # Verify result is dict-like with expected keys
    assert isinstance(result, dict)
    for key in ("estimate", "sarf", "alpha_prime", "method_used",
                "C", "C0", "N", "erf_adjustment"):
        assert key in result

    # Verify numerical properties
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["sarf"])
    assert result["alpha_prime"] is not None
    assert result["method_used"] == "ar6"

    # For CO2 above reference, SARF should be positive
    assert result["sarf"] > 0

    # Default erf_adjustment=False means estimate == sarf
    assert result["erf_adjustment"] is False
    assert math.isclose(result["estimate"], result["sarf"], rel_tol=1e-12)

    # Echoed inputs
    assert result["C"] == float(C)


def test_co2RF_edge():
    """Test edge cases: erf_adjustment flag and Myhre 1998 method."""
    C = 556.6

    # AR6 + 5% ERF adjustment
    result_erf = radiative_forcing_co2(C, erf_adjustment=True)
    result_sarf = radiative_forcing_co2(C, erf_adjustment=False)

    assert isinstance(result_erf, dict)
    assert result_erf["erf_adjustment"] is True
    assert result_sarf["erf_adjustment"] is False
    assert math.isclose(result_erf["estimate"],
                        1.05 * result_sarf["estimate"],
                        rel_tol=1e-12)
    assert math.isclose(result_erf["estimate"],
                        1.05 * result_erf["sarf"],
                        rel_tol=1e-12)

    # Myhre 1998 expression: SARF = 5.35 * ln(C / C0)
    C0 = 277.15
    result_myhre = radiative_forcing_co2(C, C0=C0, method="myhre1998")
    expected_myhre = 5.35 * math.log(C / C0)
    assert result_myhre["method_used"] == "myhre1998"
    assert math.isclose(result_myhre["sarf"], expected_myhre, rel_tol=1e-12)
    assert result_myhre["alpha_prime"] is None
    assert math.isclose(result_myhre["estimate"], expected_myhre, rel_tol=1e-12)
