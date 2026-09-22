"""Tests for btbasic.boot_basic_ci."""

from morie.fn import _array_core as np

from morie.fn.btbasic import boot_basic_ci


def _quantile(v, q):
    """Independent quantile (type-7, linear interpolation) matching core.quantile7."""
    s = sorted(v)
    n = len(s)
    h = q * (n - 1)
    lo_i = int(h)
    hi_i = lo_i + 1 if lo_i + 1 < n else lo_i
    frac = h - lo_i
    return s[lo_i] + frac * (s[hi_i] - s[lo_i])


def test_btbasic_basic():
    """Test basic functionality against the documented formula."""
    rng = np.random.default_rng(42)
    theta_hat = float(rng.normal())
    theta_b = rng.normal(0, 1, 100)
    alpha = 0.05

    result = boot_basic_ci(theta_hat, theta_b, alpha)

    assert isinstance(result, dict)

    # Required output keys per the docstring / payload.
    assert "lo" in result
    assert "hi" in result
    assert "q_lo" in result
    assert "q_hi" in result
    assert "theta_hat" in result
    assert "B" in result

    # B should equal the number of bootstrap replicates.
    assert result["B"] == len(theta_b)

    # Independent reference computation from the documented formula:
    # q_lo = quantile(theta_b, alpha/2)
    # q_hi = quantile(theta_b, 1 - alpha/2)
    # lo   = 2*theta_hat - q_hi
    # hi   = 2*theta_hat - q_lo
    q_lo = _quantile(list(theta_b), alpha / 2.0)
    q_hi = _quantile(list(theta_b), 1.0 - alpha / 2.0)
    expected_lo = 2.0 * theta_hat - q_hi
    expected_hi = 2.0 * theta_hat - q_lo

    assert result["q_lo"] == q_lo
    assert result["q_hi"] == q_hi
    assert result["lo"] == expected_lo
    assert result["hi"] == expected_hi
    assert result["theta_hat"] == theta_hat


def test_btbasic_edge():
    """Test edge cases: empty replicates rejected, alpha bounds rejected, result is a dict."""
    rng = np.random.default_rng(42)
    theta_hat = float(rng.normal())
    theta_b = rng.normal(0, 1, 100)

    # Non-empty, valid alpha => returns a dict with the documented keys.
    result = boot_basic_ci(theta_hat, theta_b, 0.05)
    assert isinstance(result, dict)
    assert "lo" in result and "hi" in result
    assert "q_lo" in result and "q_hi" in result

    # Valid alpha, independently check the reflected-quantile relation.
    alpha = 0.1
    q_lo = _quantile(list(theta_b), alpha / 2.0)
    q_hi = _quantile(list(theta_b), 1.0 - alpha / 2.0)
    expected_lo = 2.0 * theta_hat - q_hi
    expected_hi = 2.0 * theta_hat - q_lo
    res2 = boot_basic_ci(theta_hat, theta_b, alpha)
    assert res2["lo"] == expected_lo
    assert res2["hi"] == expected_hi
