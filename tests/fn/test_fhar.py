"""Tests for fhar.fourier_basis."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.fhar import fourier_basis


def test_fhar_basic():
    """Test basic functionality."""
    t = np.linspace(0.0, 10.0, 100)
    K = 3
    result = fourier_basis(t, K)
    # The function returns a RichResult object; its dict-like payload has keys:
    # 'estimate', 'Phi', 'omega', 'period', 'n', 'nbasis', 'method'.
    payload = result.payload
    assert "estimate" in payload
    assert "Phi" in payload
    assert "omega" in payload
    assert "period" in payload

    # estimate is Phi[0][0], the constant basis value, always 1
    assert payload["estimate"] == 1.0

    # The default period is the range of t = max(t) - min(t)
    P = float(np.max(t)) - float(np.min(t))
    omega = 2.0 * np.pi / P

    # Phi has shape (n, 2K+1)
    n = 100
    expected_cols = 2 * K + 1
    assert len(payload["Phi"]) == n
    assert len(payload["Phi"][0]) == expected_cols
    assert payload["n"] == n
    assert payload["nbasis"] == expected_cols

    # Spot-check a few Phi entries against the documented formula:
    #   row 0 is the constant 1 followed by sin(r w t_i), cos(r w t_i) for r=1..K.
    # Row 0, column 1 should be sin(1 * omega * t[0])
    t_first = float(t[0])
    assert abs(payload["Phi"][0][1] - np.sin(1 * omega * t_first)) < 1e-12
    # Row 0, column 2 should be cos(1 * omega * t[0])
    assert abs(payload["Phi"][0][2] - np.cos(1 * omega * t_first)) < 1e-12

    # Last row, last column should be cos(K * omega * t[-1])
    t_last = float(t[-1])
    last_idx = expected_cols - 1
    assert abs(payload["Phi"][-1][last_idx] - np.cos(K * omega * t_last)) < 1e-12

    # omega and period match the documented default
    assert abs(payload["omega"] - omega) < 1e-12
    assert abs(payload["period"] - P) < 1e-12


def test_fhar_explicit_period():
    """Test with an explicit period value."""
    t = np.linspace(0.0, 10.0, 50)
    K = 2
    period = 5.0
    result = fourier_basis(t, K, period=period)
    payload = result.payload

    P = float(period)
    omega = 2.0 * np.pi / P

    assert abs(payload["omega"] - omega) < 1e-12
    assert abs(payload["period"] - P) < 1e-12

    n = 50
    expected_cols = 2 * K + 1
    assert len(payload["Phi"]) == n
    assert len(payload["Phi"][0]) == expected_cols

    # Verify the sin/cos entries match the formula for a generic interior row
    i = 10
    ti = float(t[i])
    # Column layout: 0 -> const 1; then for r=1..K: 2r-1 -> sin, 2r -> cos
    for r in range(1, K + 1):
        s_idx = 2 * r - 1
        c_idx = 2 * r
        assert abs(payload["Phi"][i][s_idx] - np.sin(r * omega * ti)) < 1e-12
        assert abs(payload["Phi"][i][c_idx] - np.cos(r * omega * ti)) < 1e-12


def test_fhar_edge():
    """Test edge cases: K = 0 produces only the constant basis function."""
    t = np.linspace(0.0, 10.0, 100)
    K = 0
    result = fourier_basis(t, K)
    payload = result.payload

    # With K=0, the basis has 2*0 + 1 = 1 function: the constant 1.
    assert payload["nbasis"] == 1
    assert len(payload["Phi"]) == 100
    for row in payload["Phi"]:
        assert len(row) == 1
        assert row[0] == 1.0

    # estimate is still 1
    assert payload["estimate"] == 1.0
