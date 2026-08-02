"""rng014: sigma_y^2 = sigma_x^2 + sigma_eta^2 (Rangayyan 2024, Eq. 3.14, p. 96)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.rng014 import rangayyan_ch3_variance_of_sum_uncorrelated as var_sum


def test_rng014_pythagorean_triple_is_exact():
    """sigma_x=3, sigma_eta=4 -> variance 25, sd 5. Exact in binary floating point."""
    r = var_sum(3.0, 4.0)
    assert r["variance"] == 25.0
    assert r["sd"] == 5.0


def test_rng014_matches_a_simulated_uncorrelated_sum():
    """The identity is a claim about data; check it against data.

    Two independent processes are generated with known SDs, summed, and the
    sample variance of the sum compared with the formula. This is the check
    that would fail if the code returned, say, (sigma_x + sigma_eta)**2 --
    which agrees with the truth in no case except when one SD is zero.
    """
    rng = np.random.default_rng(20260726)
    n = 400_000
    sx, se = 3.0, 4.0
    x = rng.standard_normal(n) * sx
    eta = rng.standard_normal(n) * se
    empirical = float(np.var(x + eta))
    predicted = var_sum(sx, se)["variance"]
    assert predicted == pytest.approx(empirical, rel=0.01)
    # The naive wrong form is decisively excluded at this sample size.
    assert (sx + se) ** 2 != pytest.approx(empirical, rel=0.01)


def test_rng014_is_symmetric_and_additive_in_variance():
    a = var_sum(1.5, 2.5)["variance"]
    b = var_sum(2.5, 1.5)["variance"]
    assert a == b == pytest.approx(1.5**2 + 2.5**2)


def test_rng014_zero_noise_returns_the_signal_variance():
    r = var_sum(2.0, 0.0)
    assert r["variance"] == 4.0
    assert r["sd"] == 2.0


def test_rng014_broadcasts_over_arrays():
    r = var_sum(np.array([3.0, 5.0]), np.array([4.0, 12.0]))
    assert np.allclose(r["variance"], [25.0, 169.0])
    assert np.allclose(r["sd"], [5.0, 13.0])


def test_rng014_rejects_negative_sd():
    """A negative SD is a caller error -- squaring it would hide the mistake."""
    with pytest.raises(ValueError, match="non-negative"):
        var_sum(-1.0, 2.0)


def test_rng014_rejects_non_finite():
    with pytest.raises(ValueError, match="finite"):
        var_sum(np.nan, 2.0)
