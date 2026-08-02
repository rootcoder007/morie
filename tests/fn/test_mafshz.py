"""mafshz: Fisher's z transform (Fisher 1921)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.mafshi import ma_fishers_z_inverse as z_to_r
from morie.fn.mafshz import ma_fishers_z as r_to_z


def test_mafshz_matches_the_printed_closed_form():
    """z = 0.5 ln((1+r)/(1-r)), evaluated independently of arctanh."""
    for r in (-0.9, -0.3, 0.0, 0.25, 0.5, 0.99):
        expected = 0.5 * np.log((1 + r) / (1 - r))
        assert r_to_z(r)["z"] == pytest.approx(expected, rel=1e-14)


def test_mafshz_known_values():
    assert r_to_z(0.0)["z"] == 0.0
    # r = 0.5 -> 0.5*ln(3) = 0.5493061443340548
    assert r_to_z(0.5)["z"] == pytest.approx(0.5493061443340548, rel=1e-14)


def test_mafshz_variance_is_one_over_n_minus_three():
    r = r_to_z(0.4, n=103)
    assert r["var"] == pytest.approx(1.0 / 100.0)
    assert r["se"] == pytest.approx(0.1)


def test_mafshz_round_trips_through_the_inverse():
    rng = np.random.default_rng(61)
    for r in rng.uniform(-0.999, 0.999, 200):
        assert z_to_r(r_to_z(r)["z"])["r"] == pytest.approx(r, abs=1e-12)


def test_mafshz_is_odd_and_monotone():
    assert r_to_z(0.7)["z"] == pytest.approx(-r_to_z(-0.7)["z"])
    zs = [r_to_z(r)["z"] for r in np.linspace(-0.99, 0.99, 50)]
    assert zs == sorted(zs)


def test_mafshz_stabilises_the_variance():
    """The point of the transform: Var(z) stops depending on rho.

    Simulate at two very different true correlations and check that the SD of
    z is close to 1/sqrt(n-3) in both, while the SD of raw r is not constant.
    """
    rng = np.random.default_rng(71)
    n, reps = 25, 800

    def sd_of(rho, transform):
        out = []
        for _ in range(reps):
            a = rng.standard_normal(n)
            b = rho * a + np.sqrt(1 - rho**2) * rng.standard_normal(n)
            r = float(np.corrcoef(a, b)[0, 1])
            out.append(r_to_z(r)["z"] if transform else r)
        return float(np.std(out))

    target = 1.0 / np.sqrt(n - 3)
    assert sd_of(0.1, True) == pytest.approx(target, rel=0.25)
    assert sd_of(0.9, True) == pytest.approx(target, rel=0.25)
    # Untransformed, the SD collapses as rho approaches 1.
    assert sd_of(0.9, False) < 0.5 * sd_of(0.1, False)


def test_mafshz_rejects_boundary_correlation():
    with pytest.raises(ValueError, match=r"\|r\| < 1"):
        r_to_z(1.0)
    with pytest.raises(ValueError, match=r"\|r\| < 1"):
        r_to_z(-1.0)


def test_mafshz_rejects_n_that_makes_the_variance_meaningless():
    """n = 3 divides by zero; n < 3 gives a NEGATIVE variance."""
    with pytest.raises(ValueError, match="n must be > 3"):
        r_to_z(0.5, n=3)
    with pytest.raises(ValueError, match="n must be > 3"):
        r_to_z(0.5, n=2)
