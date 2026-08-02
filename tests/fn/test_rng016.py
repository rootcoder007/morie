"""rng016: ensemble-average ACF (Rangayyan 2024, Eq. 3.16/3.17, p. 96)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.rng016 import rangayyan_ch3_acf_continuous as acf


def test_rng016_planted_ensemble_average():
    """Hand-built ensemble: phi = mean over realisations of x_k(t1)*x_k(t1+tau).

    Column 0 is (1,2,3), column 2 is (4,5,6), so with t1=0, tau=2 the answer
    is (1*4 + 2*5 + 3*6)/3 = 32/3.
    """
    x = np.array(
        [
            [1.0, 9.0, 4.0],
            [2.0, 9.0, 5.0],
            [3.0, 9.0, 6.0],
        ]
    )
    r = acf(x, t1=0, tau=2)
    assert r["value"] == pytest.approx(32.0 / 3.0)
    assert r["M"] == 3
    assert r["n"] == 3


def test_rng016_zero_lag_is_the_mean_square():
    """tau=0 reduces Eq. (3.16) to E[x^2(t1)], the MS value of Eq. (3.2)."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((500, 8))
    got = acf(x, t1=3, tau=0)["value"]
    assert got == pytest.approx(float(np.mean(x[:, 3] ** 2)))


def test_rng016_recovers_a_known_process_acf():
    """White noise of variance s^2: phi(0)=s^2 and phi(tau!=0)=0.

    Ground truth is analytic, so this checks correctness, not just internal
    consistency.
    """
    rng = np.random.default_rng(11)
    s = 2.0
    x = rng.standard_normal((200_000, 4)) * s
    assert acf(x, t1=1, tau=0)["value"] == pytest.approx(s**2, rel=0.02)
    assert acf(x, t1=1, tau=1)["value"] == pytest.approx(0.0, abs=0.02)


def test_rng016_negative_lag_is_symmetric():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((300, 6))
    forward = acf(x, t1=1, tau=3)["value"]
    backward = acf(x, t1=4, tau=-3)["value"]
    assert forward == pytest.approx(backward)


def test_rng016_rejects_a_single_realisation():
    """A 1-D input would silently compute the TIME average of Eq. (3.20).

    Those coincide only for an ergodic process, so the substitution has to be
    the caller's explicit choice, not a reshape this function performs.
    """
    with pytest.raises(ValueError, match="2-D ensemble"):
        acf(np.arange(10.0), t1=0, tau=1)


def test_rng016_rejects_out_of_range_lag():
    x = np.zeros((4, 5))
    with pytest.raises(ValueError, match="out of range"):
        acf(x, t1=3, tau=5)
