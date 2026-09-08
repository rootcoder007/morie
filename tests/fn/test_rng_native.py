"""Tests for the native RNG: Philox4x32-10 and Wichura's AS 241.

Correctness is established against PUBLISHED Known Answer Tests, not against
whatever the implementation happens to produce.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn._rng import (normal_quantile, philox4x32, random_multivariate_normal,
                           random_normal, random_uniform)

# Known Answer Tests from the Random123 reference distribution accompanying
# Salmon, Moraes, Dror & Shaw (2011). The third counter/key pair is the
# leading hex digits of pi, which is the suite's standard vector.
PHILOX_KAT = [
    ((0, 0, 0, 0), (0, 0),
     (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    ((0xFFFFFFFF,) * 4, (0xFFFFFFFF,) * 2,
     (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
    ((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0),
     (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)),
]


@pytest.mark.parametrize("ctr,key,want", PHILOX_KAT)
def test_philox_matches_the_published_known_answer_tests(ctr, key, want):
    assert philox4x32(ctr, key) == want


def test_as241_matches_published_normal_quantiles():
    """Wichura (1988) is accurate to about 1e-16; these are standard table
    values."""
    cases = {0.975: 1.959963984540054, 0.5: 0.0,
             0.001: -3.090232306167813, 0.99: 2.3263478740408408,
             0.025: -1.959963984540054}
    for p, want in cases.items():
        assert float(normal_quantile(np.array([p]))[0]) == pytest.approx(want, abs=1e-13)


def test_normal_quantile_is_antisymmetric():
    """Phi^-1(1 - p) = -Phi^-1(p) exactly in the maths; an asymmetric
    implementation would bias every simulation built on it.

    Checked only where 1 - p is representable without cancellation. Below
    about p = 1e-9 forming 1 - p loses the low digits outright: for p = 1e-8,
    1 - p rounds to 0.99999999 and the tail recovered inside AS 241 is
    9.99999993923e-9, an eight-digit loss worth ~1e-9 in Phi^-1. That is
    double precision, not the algorithm. Note also that np.allclose carries a
    default RELATIVE tolerance of 1e-5, which would have hidden this -- the
    comparison below is absolute.
    """
    p = np.array([0.001, 0.01, 0.1, 0.3, 0.49])
    got = normal_quantile(1.0 - p)
    want = -normal_quantile(p)
    assert np.max(np.abs(got - want)) < 1e-12


def test_extreme_upper_tail_is_limited_by_representing_one_minus_p():
    """Documented, not hidden: through an exact argument the tail is exact;
    through 1 - p it is not."""
    assert float(normal_quantile(np.array([1e-8]))[0]) == pytest.approx(
        -5.61200124417479, abs=1e-13)
    lost = abs(float(normal_quantile(np.array([1.0 - 1e-8]))[0])
               + float(normal_quantile(np.array([1e-8]))[0]))
    assert lost < 1e-8


def test_uniforms_never_reach_the_endpoints():
    """A normal quantile at 0 or 1 is infinite, so the open interval is a
    correctness requirement, not a nicety."""
    u = random_uniform(100000, seed=7)
    assert u.min() > 0.0
    assert u.max() < 1.0
    assert np.all(np.isfinite(normal_quantile(u)))


def test_moments_are_right():
    z = random_normal(200000, seed=42)
    assert z.mean() == pytest.approx(0.0, abs=0.01)
    assert z.std(ddof=1) == pytest.approx(1.0, abs=0.01)
    skew = float(((z - z.mean()) ** 3).mean() / z.std() ** 3)
    kurt = float(((z - z.mean()) ** 4).mean() / z.std() ** 4)
    assert skew == pytest.approx(0.0, abs=0.05)
    assert kurt == pytest.approx(3.0, abs=0.05)


def test_streams_and_seeds_are_independent_handles():
    a = random_uniform(1000, seed=1, stream=0)
    b = random_uniform(1000, seed=1, stream=1)
    c = random_uniform(1000, seed=2, stream=0)
    assert not np.allclose(a, b)
    assert not np.allclose(a, c)
    assert np.allclose(a, random_uniform(1000, seed=1, stream=0))


def test_counter_based_means_any_offset_is_reachable():
    """Philox is a bijection of the index, so a long draw must contain the
    short one as a prefix -- there is no state to wind forward."""
    long_draw = random_uniform(64, seed=99)
    assert np.array_equal(long_draw[:9], random_uniform(9, seed=99))


def test_agrees_with_the_r_arm_bit_for_bit():
    """The whole point of an integer-only counter-based generator: the R
    twin produces these same doubles, not merely the same distribution."""
    u = random_uniform(7, seed=12345, stream=3)
    z = random_normal(7, seed=12345, stream=3)
    assert u[0] == pytest.approx(0.82723027456086129, rel=0, abs=0)
    assert u[6] == pytest.approx(0.36555732542183250, rel=0, abs=0)
    assert z[0] == pytest.approx(0.94327658191243779, rel=0, abs=0)
    assert z[2] == pytest.approx(-1.19034287143374250, rel=0, abs=0)


def test_multivariate_normal_reproduces_the_target_covariance():
    """The Cholesky construction Schabenberger & Gotway use for simulating a
    Gaussian field: over many draws the sample covariance must approach the
    one asked for."""
    cov = np.array([[2.0, 0.8, 0.3], [0.8, 1.5, 0.2], [0.3, 0.2, 1.0]])
    mean = np.array([1.0, -2.0, 0.5])
    draws = np.array([random_multivariate_normal(mean, cov, seed=5, stream=s)
                      for s in range(4000)])
    assert np.allclose(draws.mean(axis=0), mean, atol=0.1)
    assert np.allclose(np.cov(draws.T), cov, atol=0.12)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        random_uniform(-1)
    with pytest.raises(ValueError):
        normal_quantile(np.array([0.0]))
    with pytest.raises(ValueError):
        normal_quantile(np.array([1.0]))
