"""totcorr: total correlation / multi-information (Watanabe 1960)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.totcorr import total_correlation as tc


def test_totcorr_independent_variables_give_exactly_zero():
    """TC = 0 iff mutually independent -- it is KL(joint || product of marginals)."""
    p = np.outer([0.3, 0.7], [0.4, 0.6])
    assert tc(p)["estimate"] == pytest.approx(0.0, abs=1e-15)


def test_totcorr_two_variables_equal_the_mutual_information():
    """For n = 2, TC(X1,X2) = H(X1) + H(X2) - H(X1,X2) = I(X1;X2)."""
    p = np.array([[0.1, 0.2], [0.3, 0.4]])
    px, py = p.sum(1), p.sum(0)
    mi = float(
        sum(
            p[i, j] * np.log2(p[i, j] / (px[i] * py[j]))
            for i in range(2)
            for j in range(2)
        )
    )
    assert tc(p)["estimate"] == pytest.approx(mi, rel=1e-12)


def test_totcorr_two_perfectly_coupled_fair_bits_give_one_bit():
    """X1 = X2 uniform: H(X1)=H(X2)=1, H(X1,X2)=1, so TC = 1 bit."""
    p = np.array([[0.5, 0.0], [0.0, 0.5]])
    assert tc(p)["estimate"] == pytest.approx(1.0)


def test_totcorr_three_copies_of_a_fair_bit_give_two_bits():
    """Three identical fair bits: 3*1 - 1 = 2 bits."""
    p = np.zeros((2, 2, 2))
    p[0, 0, 0] = p[1, 1, 1] = 0.5
    assert tc(p)["estimate"] == pytest.approx(2.0)


def test_totcorr_is_non_negative_on_random_distributions():
    rng = np.random.default_rng(97)
    for _ in range(200):
        p = rng.random((3, 2, 4))
        p /= p.sum()
        assert tc(p)["estimate"] >= -1e-12


def test_totcorr_nats_are_bits_times_ln_two():
    p = np.array([[0.5, 0.0], [0.0, 0.5]])
    assert tc(p, base="nats")["estimate"] == pytest.approx(
        tc(p, base="bits")["estimate"] * np.log(2.0)
    )


def test_totcorr_rejects_a_one_dimensional_input():
    """A marginal is not a joint; TC of one variable is identically 0."""
    with pytest.raises(ValueError, match="JOINT PMF"):
        tc(np.array([0.5, 0.5]))


def test_totcorr_rejects_an_unnormalised_or_negative_pmf():
    with pytest.raises(ValueError, match="sum to 1"):
        tc(np.array([[0.5, 0.5], [0.5, 0.5]]))
    with pytest.raises(ValueError, match="non-negative"):
        tc(np.array([[-0.5, 0.5], [0.5, 0.5]]))
