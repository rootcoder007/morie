"""dualtc: dual total correlation (Han 1978)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.dualtc import dual_total_correlation as dtc
from morie.fn.totcorr import total_correlation as tc


def test_dualtc_independent_variables_give_exactly_zero():
    p = np.outer([0.3, 0.7], [0.4, 0.6])
    assert dtc(p)["estimate"] == pytest.approx(0.0, abs=1e-15)


def test_dualtc_equals_total_correlation_only_for_two_variables():
    """At n = 2 both reduce to the mutual information."""
    rng = np.random.default_rng(103)
    for _ in range(30):
        p = rng.random((3, 4))
        p /= p.sum()
        assert dtc(p)["estimate"] == pytest.approx(tc(p)["estimate"], rel=1e-10)


def test_dualtc_differs_from_total_correlation_for_three_variables():
    """They share a zero set but are different quantities for n >= 3.

    Three copies of a fair bit: TC = 2 bits, DTC = 1 bit. A test that only
    checked non-negativity would not distinguish the two functions at all.
    """
    p = np.zeros((2, 2, 2))
    p[0, 0, 0] = p[1, 1, 1] = 0.5
    assert tc(p)["estimate"] == pytest.approx(2.0)
    assert dtc(p)["estimate"] == pytest.approx(1.0)


def test_dualtc_xor_has_zero_pairwise_but_positive_dtc():
    """X3 = X1 xor X2 with X1, X2 independent fair bits.

    Every pair is independent, so all pairwise mutual informations vanish,
    yet the triple is completely determined. DTC = 2 bits.
    """
    p = np.zeros((2, 2, 2))
    for a in (0, 1):
        for b in (0, 1):
            p[a, b, a ^ b] = 0.25
    assert dtc(p)["estimate"] == pytest.approx(2.0)
    # Pairwise independence: each 2-way marginal is uniform.
    for axis in range(3):
        assert np.allclose(p.sum(axis=axis), 0.25)


def test_dualtc_is_non_negative_on_random_distributions():
    rng = np.random.default_rng(107)
    for _ in range(200):
        p = rng.random((2, 3, 2))
        p /= p.sum()
        assert dtc(p)["estimate"] >= -1e-12


def test_dualtc_nats_are_bits_times_ln_two():
    p = np.zeros((2, 2, 2))
    p[0, 0, 0] = p[1, 1, 1] = 0.5
    assert dtc(p, base="nats")["estimate"] == pytest.approx(
        dtc(p, base="bits")["estimate"] * np.log(2.0)
    )


def test_dualtc_rejects_bad_input():
    with pytest.raises(ValueError, match="JOINT PMF"):
        dtc(np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="sum to 1"):
        dtc(np.ones((2, 2)))
