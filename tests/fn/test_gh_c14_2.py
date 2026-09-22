"""Tests for gh_c14_2.ghosal_ewens_esf."""

import math

from morie.fn import _array_core as np

from morie.fn.gh_c14_2 import ghosal_ewens_esf


def _expected_esf(ms, alpha=1.0):
    """Independent reference: Ewens sampling formula, Prop 4.10.

    P = n! / alpha^{[n]} * prod_{i} ( alpha^{m_i} / (i^{m_i} m_i!) )
    where n = sum_i i*m_i (i ranges over block sizes 1..len(ms)).
    """
    ms = [int(v) for v in ms]
    n = sum((i + 1) * m for i, m in enumerate(ms))
    log_asc = sum(math.log(alpha + i) for i in range(n))
    lp = math.lgamma(n + 1.0) - log_asc
    for i, m in enumerate(ms):
        size = i + 1
        lp += m * math.log(alpha) - m * math.log(size) \
            - math.lgamma(m + 1.0)
    return math.exp(lp), lp, n


def test_gh_c14_2_basic():
    """Test basic functionality with a valid multiplicity class.

    multiplicities = [1, 2, 3, 4, 5] means:
      m_1=1 (one block of size 1), m_2=2 (two blocks of size 2),
      m_3=3, m_4=4, m_5=5.
    n = 1*1 + 2*2 + 3*3 + 4*4 + 5*5 = 55.
    """
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_ewens_esf(x)

    assert "estimate" in result
    assert "log_prob" in result
    assert "n" in result
    assert "method" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))

    # Compute expected values independently from the formula.
    exp_est, exp_lp, exp_n = _expected_esf([1, 2, 3, 4, 5])
    assert result["n"] == exp_n
    assert abs(result["log_prob"] - exp_lp) < 1e-12
    assert abs(result["estimate"] - exp_est) < 1e-12


def test_gh_c14_2_edge():
    """Test edge case: single allele (one block of size 1).

    multiplicities = [1] -> ms[0]=1, size=1, so n = 1*1 = 1.
    With alpha=1.0: P = 1! / 1^{[1]} * 1^1 / (1^1 * 1!) = 1.
    """
    result = ghosal_ewens_esf(np.array([1.0]))
    assert result["n"] == 1
    assert result["estimate"] == 1.0
    assert result["log_prob"] == 0.0

    # Independent verification.
    exp_est, exp_lp, exp_n = _expected_esf([1])
    assert exp_n == 1
    assert abs(result["estimate"] - exp_est) < 1e-12
    assert abs(result["log_prob"] - exp_lp) < 1e-12
