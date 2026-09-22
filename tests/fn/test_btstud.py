"""Tests for btstud.boot_studentized_ci."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.btstud import boot_studentized_ci


def test_btstud_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    theta_hat = float(rng.normal())
    se_hat = float(abs(rng.normal()) + 0.5)
    t_b = rng.normal(size=100)
    alpha = 0.05
    result = boot_studentized_ci(theta_hat, se_hat, t_b, alpha)

    # Function returns a RichResult; it must be dict-like and expose the documented keys.
    assert hasattr(result, "payload") or isinstance(result, dict)
    payload = result.payload if hasattr(result, "payload") else result

    v = np.asarray(t_b)
    a = float(alpha)
    s = float(se_hat)
    t = float(theta_hat)
    zlo = float(np.quantile(v, a / 2.0))
    zhi = float(np.quantile(v, 1.0 - a / 2.0))
    expected_lo = t - zhi * s
    expected_hi = t - zlo * s

    assert "lo" in payload
    assert "hi" in payload
    assert "estimate" in payload
    assert "z_lo" in payload
    assert "z_hi" in payload
    assert payload["lo"] == expected_lo
    assert payload["hi"] == expected_hi
    assert payload["estimate"] == expected_hi - expected_lo
    assert payload["z_lo"] == zlo
    assert payload["z_hi"] == zhi
    assert payload["se_hat"] == s
    assert payload["theta_hat"] == t
    assert payload["B"] == len(v)


def test_btstud_edge():
    """Test edge cases: array-like t_b and the documented dict-like return shape."""
    rng = np.random.default_rng(42)
    theta_hat = float(rng.normal())
    se_hat = float(abs(rng.normal()) + 0.5)
    t_b = list(rng.normal(size=50))  # exercise the array-like path explicitly
    alpha = 0.05
    result = boot_studentized_ci(theta_hat, se_hat, t_b, alpha)

    payload = result.payload if hasattr(result, "payload") else result
    assert isinstance(payload, dict)
    assert "lo" in payload
    assert "hi" in payload
    assert "estimate" in payload
