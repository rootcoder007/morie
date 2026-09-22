"""Tests for gb_wss.gibbons_wrs_normal_approx."""

from morie.fn import _array_core as np

from morie.fn.gb_wss import gibbons_wrs_normal_approx


def test_gb_wss_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    m, n = 10, 100
    N = m + n
    # W is the sum of ranks of the m-sample (a single scalar).
    w = rng.uniform(0.0, N * (N + 1) / 2.0)
    result = gibbons_wrs_normal_approx(w, m, n)
    assert isinstance(result, dict)
    assert "z" in result
    assert "p_value" in result
    assert "mean" in result
    assert "var" in result
    assert "var_uncorrected" in result
    assert result["m"] == m
    assert result["n"] == n

    # Independent recomputation from the documented formula.
    mean = m * (N + 1) / 2.0
    var = m * n * (N + 1) / 12.0
    expected_z = (float(w) - mean) / np.sqrt(var)
    expected_p = 2.0 * (1.0 - 0.5 * (1.0 + _erf(abs(expected_z) / np.sqrt(2.0))))
    assert abs(result["mean"] - mean) < 1e-12
    assert abs(result["var"] - var) < 1e-12
    assert abs(result["var_uncorrected"] - var) < 1e-12
    assert abs(result["z"] - expected_z) < 1e-12
    assert abs(result["p_value"] - min(1.0, expected_p)) < 1e-12


def test_gb_wss_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    m, n = 1, 1
    N = m + n
    w = 1.0  # minimum possible W for m=n=1
    result = gibbons_wrs_normal_approx(w, m, n)
    assert isinstance(result, dict)
    assert "z" in result and "p_value" in result
    assert result["m"] == m and result["n"] == n

    # Independent recomputation.
    mean = m * (N + 1) / 2.0
    var = m * n * (N + 1) / 12.0
    expected_z = (w - mean) / np.sqrt(var)
    assert abs(result["mean"] - mean) < 1e-12
    assert abs(result["var"] - var) < 1e-12
    assert abs(result["z"] - expected_z) < 1e-12


def _erf(x):
    # Abramowitz & Stegun 7.1.26 approximation of erf, used only inside the
    # test to compute a normal-CDF value without depending on scipy/statsmodels.
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y
