"""Tests for co2RF.radiative_forcing_co2."""

from morie.fn import _array_core as np

from morie.fn.co2RF import radiative_forcing_co2


def test_co2RF_basic():
    """Test basic functionality (AR6 method, default reference values)."""
    # Use a single-element marr so float() coercion works, mirroring the
    # documented "float" parameter contract.
    C = float(np.asarray([556.6])[0])   # ~2x pre-industrial CO2
    result = radiative_forcing_co2(C)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "sarf" in result
    assert "alpha_prime" in result
    assert "method_used" in result

    # Independent recomputation from the AR6 formula, using the same inputs.
    import math
    C0 = 277.15
    N = 273.87
    _A1 = 0.0   # placeholder; overwritten from module constants below
    _B1 = 0.0
    _C1 = 0.0
    _D1 = 0.0
    # Pull the fitted constants from the function's closure / module.
    import morie.fn.co2RF as _co2mod
    A1, B1, C1, D1 = _co2mod._A1, _co2mod._B1, _co2mod._C1, _co2mod._D1
    c_amax = C0 - B1 / (2.0 * A1)
    if C > c_amax:
        alpha = D1 - B1 * B1 / (4.0 * A1)
    elif C > C0:
        alpha = D1 + A1 * (C - C0) ** 2 + B1 * (C - C0)
    else:
        alpha = D1
    expected_sarf = (alpha + C1 * math.sqrt(N)) * math.log(C / C0)

    assert math.isclose(result["sarf"], expected_sarf, rel_tol=1e-12)
    assert math.isclose(result["estimate"], expected_sarf, rel_tol=1e-12)
    assert result["method_used"] == "ar6"


def test_co2RF_edge():
    """Test edge cases: erf_adjustment flag and Myhre 1998 method."""
    import math

    C = 556.6
    # AR6 + 5% ERF adjustment, evaluated against the plain SARF baseline.
    result_erf = radiative_forcing_co2(C, erf_adjustment=True)
    result_sarf = radiative_forcing_co2(C, erf_adjustment=False)
    assert isinstance(result_erf, dict)
    assert result_erf["erf_adjustment"] is True
    assert result_sarf["erf_adjustment"] is False
    assert math.isclose(result_erf["estimate"],
                        1.05 * result_sarf["estimate"],
                        rel_tol=1e-12)

    # Myhre 1998 expression: SARF = 5.35 * ln(C / C0).
    C0 = 277.15
    result_myhre = radiative_forcing_co2(C, C0=C0, method="myhre1998")
    expected_myhre = 5.35 * math.log(C / C0)
    assert result_myhre["method_used"] == "myhre1998"
    assert math.isclose(result_myhre["sarf"], expected_myhre, rel_tol=1e-12)
    assert result_myhre["alpha_prime"] is None
