"""Tests for fzt14.fauzi_thm1_4_asympnorm_mgkde."""

from morie.fn import _array_core as np
from morie.fn.fzt14 import fauzi_thm1_4_asympnorm_mgkde


def test_fzt14_basic():
    """Test basic functionality (interior, closed form with f)."""
    x = 1.5
    n = 200
    h = 0.04
    f = 0.3
    result = fauzi_thm1_4_asympnorm_mgkde(x, h, n, f=f)
    assert isinstance(result, dict)
    assert "covariance" in result
    assert result["form"] == "interior"
    assert result["method"].startswith("Cov[A_h, A_4h]")
    assert result["h"] == h
    assert result["n"] == n

    # Independent recomputation of the Theorem 1.4 interior closed form.
    s = np.sqrt(h)
    den = 3.0 * x + 5.0 * s
    num1 = x + s
    num2 = 2.0 * x + 4.0 * s
    from morie.fn._fauzi import rratio
    r1 = float(np.atleast_1d(rratio(1.0 / s - 1.0))[0])
    r2 = float(np.atleast_1d(rratio(1.0 / (2.0 * s) - 1.0))[0])
    r3 = float(np.atleast_1d(rratio(3.0 / (2.0 * s) - 2.0))[0])
    log_v = (
        np.log(r1)
        + np.log(r2)
        - np.log(r3)
        + (3.0 / (2.0 * s) - 1.5) * np.log(1.5 - 2.0 * s)
        - 0.5 * np.log(np.pi)
        - np.log(2.0)
        - np.log(den)
        - (1.0 / s - 0.5) * np.log(2.0 - 2.0 * s)
        - (1.0 / (2.0 * s) - 0.5) * np.log(1.0 - 2.0 * s)
        + (1.0 / (2.0 * s) - 1.0) * np.log(num1 / den)
        + (1.0 / s - 1.0) * np.log(num2 / den)
    )
    expected = float(np.exp(log_v)) * f / (n * h ** 0.25)
    assert abs(result["covariance"] - expected) < 1e-10 * max(1.0, abs(expected))


def test_fzt14_edge():
    """Test edge cases (boundary branch)."""
    x = 0.0
    n = 100
    h = 0.04
    c = 1.5
    f = 0.3
    result = fauzi_thm1_4_asympnorm_mgkde(x, h, n, f=f, boundary=True, c=c)
    assert isinstance(result, dict)
    assert result["form"] == "boundary"
    assert result["covariance"] > 0

    # Independent recomputation of the Theorem 1.4 boundary closed form.
    s = np.sqrt(h)
    cc = float(c)
    den = 3.0 * cc * s + 5.0
    num1 = cc * s + 1.0
    num2 = 2.0 * cc * s + 4.0
    from morie.fn._fauzi import rratio
    r1 = float(np.atleast_1d(rratio(1.0 / s - 1.0))[0])
    r2 = float(np.atleast_1d(rratio(1.0 / (2.0 * s) - 1.0))[0])
    r3 = float(np.atleast_1d(rratio(3.0 / (2.0 * s) - 2.0))[0])
    log_v = (
        np.log(r1)
        + np.log(r2)
        - np.log(r3)
        + (3.0 / (2.0 * s) - 1.5) * np.log(1.5 - 2.0 * s)
        - 0.5 * np.log(np.pi)
        - np.log(2.0)
        - np.log(den)
        - (1.0 / s - 0.5) * np.log(2.0 - 2.0 * s)
        - (1.0 / (2.0 * s) - 0.5) * np.log(1.0 - 2.0 * s)
        + (1.0 / (2.0 * s) - 1.0) * np.log(num1 / den)
        + (1.0 / s - 1.0) * np.log(num2 / den)
    )
    expected = float(np.exp(log_v)) * f / (n * h ** 0.75)
    assert abs(result["covariance"] - expected) < 1e-10 * max(1.0, abs(expected))
