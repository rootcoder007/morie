"""Tests for basEvap.penman_monteith."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.basEvap import penman_monteith


def test_basEvap_basic():
    """Test basic functionality with FAO-56 Example 18 anchor values."""
    T = 16.9
    R_n = 13.28
    u2 = 2.078
    VPD = 0.589
    P = 100.1
    result = penman_monteith(T, R_n, u2, VPD, G=0.0, P=P)
    assert isinstance(result, dict)
    assert "estimate" in result

    # Independent recomputation of the FAO-56 Eq. 6 formula
    es_T = 0.6108 * np.exp(17.27 * T / (T + 237.3))
    delta = 4098.0 * es_T / (T + 237.3) ** 2
    gamma = 0.665e-3 * P
    denom = delta + gamma * (1.0 + 0.34 * u2)
    rad = 0.408 * delta * (R_n - 0.0) / denom
    aero = gamma * (900.0 / (T + 273.0)) * u2 * VPD / denom
    expected_et0 = rad + aero
    expected_rad = rad
    expected_aero = aero
    expected_delta = delta
    expected_gamma = gamma

    assert abs(result["estimate"] - expected_et0) < 1e-9
    assert abs(result["radiative_term"] - expected_rad) < 1e-9
    assert abs(result["aerodynamic_term"] - expected_aero) < 1e-9
    assert abs(result["delta"] - expected_delta) < 1e-9
    assert abs(result["gamma"] - expected_gamma) < 1e-9
    assert abs(result["estimate"] - 3.88) < 0.01
    assert result["T"] == T
    assert result["R_n"] == R_n
    assert result["u2"] == u2
    assert result["VPD"] == VPD
    assert result["G"] == 0.0
    assert result["P"] == P
    assert result["method"].startswith("FAO-56 Penman-Monteith")


def test_basEvap_edge():
    """Test edge cases: defaults and validation."""
    # Defaults: G=0, P=101.3 (sea level)
    T = 20.0
    R_n = 10.0
    u2 = 1.0
    VPD = 0.5
    result = penman_monteith(T, R_n, u2, VPD)
    assert isinstance(result, dict)
    assert result["G"] == 0.0
    assert result["P"] == 101.3

    es_T = 0.6108 * np.exp(17.27 * T / (T + 237.3))
    delta = 4098.0 * es_T / (T + 237.3) ** 2
    gamma = 0.665e-3 * 101.3
    denom = delta + gamma * (1.0 + 0.34 * u2)
    rad = 0.408 * delta * (R_n - 0.0) / denom
    aero = gamma * (900.0 / (T + 273.0)) * u2 * VPD / denom
    expected_et0 = rad + aero
    assert abs(result["estimate"] - expected_et0) < 1e-9

    # Edge: u2 == 0 -> only radiative term contributes
    T2 = 25.0
    R_n2 = 8.0
    VPD2 = 1.2
    res0 = penman_monteith(T2, R_n2, 0.0, VPD2)
    es_T2 = 0.6108 * np.exp(17.27 * T2 / (T2 + 237.3))
    delta2 = 4098.0 * es_T2 / (T2 + 237.3) ** 2
    gamma2 = 0.665e-3 * 101.3
    denom2 = delta2 + gamma2 * (1.0 + 0.0)
    expected_rad_only = 0.408 * delta2 * R_n2 / denom2
    assert abs(res0["aerodynamic_term"] - 0.0) < 1e-12
    assert abs(res0["estimate"] - expected_rad_only) < 1e-12
