"""Tests for gh_c6_16.ghosal_alpha_post."""

from morie.fn import _array_core as np

from morie.fn.gh_c6_16 import ghosal_alpha_post


def test_gh_c6_16_basic():
    """Test basic functionality."""
    # successes S, number of trials n, and alpha-posterior params
    successes = 3.0
    n = 5.0
    alpha = 0.5
    a = 1.0
    b = 1.0

    result = ghosal_alpha_post(successes, n, alpha=alpha, a=a, b=b)

    # Documented return keys include "estimate"; "n" is not a returned key.
    assert "estimate" in result

    est = float(np.asarray(result["estimate"], dtype=float))
    assert np.isfinite(est)

    # Independently compute the documented alpha-posterior mean:
    # Beta(a + alpha*S, b + alpha*(n-S)) => mean = A / (A + B)
    A = a + alpha * successes
    B = b + alpha * (n - successes)
    expected_mean = A / (A + B)
    assert abs(est - expected_mean) < 1e-12

    # And the alpha-posterior parameters themselves are returned.
    assert "alpha_posterior" in result
    a_post_r, b_post_r = result["alpha_posterior"]
    assert abs(float(a_post_r) - A) < 1e-12
    assert abs(float(b_post_r) - B) < 1e-12

    # The alpha-posterior variance must be >= the full-data variance
    # for 0 < alpha < 1 (documented: "flatter than the full posterior").
    assert result["variance"] > 0.0
    full_var = (a + successes) * (b + n - successes) / ((a + b + n) ** 2
                                                       * (a + b + n + 1.0))
    assert result["variance"] > full_var
    assert result["wider_than_full"] is True


def test_gh_c6_16_edge():
    """Test edge cases."""
    # Single trial with a success: S=n=1, default alpha=0.5.
    successes = 1.0
    n = 1.0
    result = ghosal_alpha_post(successes, n)

    # No "n" key is documented; check the documented keys.
    assert "estimate" in result
    assert "alpha_posterior" in result

    # With S=n, the alpha-posterior mean equals (a + alpha) / (a + b + alpha).
    alpha = 0.5
    a, b = 1.0, 1.0
    A = a + alpha * successes
    B = b + alpha * (n - successes)
    expected = A / (A + B)
    assert abs(float(np.asarray(result["estimate"], dtype=float))) - expected < 1e-12

    # The returned alpha-posterior parameters match the formula.
    assert abs(float(result["alpha_posterior"][0]) - A) < 1e-12
    assert abs(float(result["alpha_posterior"][1]) - B) < 1e-12
