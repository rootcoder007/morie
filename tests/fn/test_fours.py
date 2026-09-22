"""Tests for fours.fourier_basis."""

import math

from morie.fn import _array_core as np

from morie.fn.fours import fourier_basis


def test_fours_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    n_harmonics = 3
    result = fourier_basis(t, n_harmonics)

    # The function returns a dict-like RichResult
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "F" in result
    assert "omega" in result
    assert "period" in result

    # Shape of the basis matrix: (len(t), 2*n_harmonics+1)
    n = len(t)
    expected_cols = 2 * n_harmonics + 1
    assert len(result["F"]) == n
    for row in result["F"]:
        assert len(row) == expected_cols

    # Period defaults to range of t (Section 14.2.1)
    expected_period = max(t) - min(t)
    expected_omega = 2.0 * math.pi / expected_period
    expected_c0 = 1.0 / math.sqrt(expected_period)
    expected_ck = 1.0 / math.sqrt(expected_period / 2.0)

    assert result["period"] == expected_period
    assert result["omega"] == expected_omega

    # estimate is the first constant basis value evaluated at t[0]
    assert result["estimate"] == expected_c0

    # Spot-check a few basis rows against the independent formula
    for i in [0, 25, 50, 75, 99]:
        ti = t[i]
        expected_row = [expected_c0]
        for h in range(1, n_harmonics + 1):
            expected_row.append(expected_ck * math.sin(h * expected_omega * ti))
            expected_row.append(expected_ck * math.cos(h * expected_omega * ti))
        assert len(result["F"][i]) == len(expected_row)
        for a, b in zip(result["F"][i], expected_row):
            assert abs(a - b) < 1e-12


def test_fours_edge():
    """Test edge cases: explicit period."""
    t = np.linspace(0, 10, 100)
    n_harmonics = 2
    period = 5.0
    result = fourier_basis(t, n_harmonics, period=period)

    assert isinstance(result, dict)
    assert result["period"] == period
    expected_omega = 2.0 * math.pi / period
    assert result["omega"] == expected_omega

    expected_cols = 2 * n_harmonics + 1
    assert len(result["F"]) == len(t)
    for row in result["F"]:
        assert len(row) == expected_cols
