"""Tests for gh_c10_2.ghosal_univ_weights."""

import math

from morie.fn import _array_core as np

from morie.fn.gh_c10_2 import ghosal_univ_weights


def test_gh_c10_2_basic():
    """Test basic functionality with documented signature."""
    n, c, K_max, eps_scale = 100, 2.0, 200, 1.0
    result = ghosal_univ_weights(n=n, c=c, K_max=K_max, eps_scale=eps_scale)

    # Key names documented in the function's payload
    assert "estimate" in result
    assert "partial_sums" in result
    assert "converges" in result
    assert "method" in result

    # estimate must be a finite scalar
    est = result["estimate"]
    assert np.all(np.isfinite(np.asarray(est, dtype=float)))

    # partial_sums is a list of length 3 (k in (10, 50, K_max))
    ps = result["partial_sums"]
    assert len(ps) == 3

    # Independently compute the partial sums from the documented formula
    # log pi_k = -c * k * log n
    log_pis = [-c * k * math.log(n) for k in range(1, K_max + 1)]
    mx = max(log_pis)
    Z = sum(math.exp(v - mx) for v in log_pis)

    total = 0.0
    expected_partials = []
    for k in range(1, K_max + 1):
        n_eps2 = eps_scale * k * math.log(n)
        total += math.exp(log_pis[k - 1] - mx - math.log(Z) + n_eps2)
        if k in (10, 50, K_max):
            expected_partials.append(total)

    # Independent computation must match the function's output
    assert math.isclose(float(est), total, rel_tol=1e-12, abs_tol=1e-12)
    for got, exp in zip(ps, expected_partials):
        assert math.isclose(float(got), exp, rel_tol=1e-12, abs_tol=1e-12)

    # Documented condition: converges iff c > eps_scale and total is finite
    assert result["converges"] is True


def test_gh_c10_2_edge():
    """Test edge cases using the documented signature."""
    n = 42
    result = ghosal_univ_weights(n=n, c=2.0, K_max=200, eps_scale=1.0)

    # 'n' is a parameter, not a key returned by the function.
    # The function returns 'estimate', 'partial_sums', 'converges', 'method'.
    assert "estimate" in result
    assert "partial_sums" in result
    assert "converges" in result
    assert "method" in result
    assert math.isfinite(float(result["estimate"]))
