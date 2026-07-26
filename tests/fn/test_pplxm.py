"""pplxm: perplexity (Jelinek et al. 1977).

    PPL = exp(-1/N * sum_i log p(x_i | x_<i))

The input is per-token LOG-PROBABILITIES (negative), not losses, and `base`
is the string "e" or "2" describing the log base of the input.
"""

import numpy as np
import pytest

from morie.fn.pplxm import perplexity_metric as pp


def test_pplxm_uniform_over_k_outcomes_has_perplexity_k():
    """The defining intuition: perplexity is an effective branching factor.

    A model assigning 1/k to every one of k outcomes is exactly as confused
    as a fair k-sided die, so PPL = k.
    """
    for k in (2, 10, 50):
        logp = np.full(200, np.log(1.0 / k))
        assert pp(logp)["value"] == pytest.approx(float(k))


def test_pplxm_a_perfect_model_has_perplexity_one():
    """log p = 0 means p = 1 on every token."""
    assert pp(np.zeros(50))["value"] == pytest.approx(1.0)


def test_pplxm_matches_the_closed_form():
    rng = np.random.default_rng(2003)
    logp = -rng.uniform(0.1, 4.0, 100)
    r = pp(logp)
    assert r["nll"] == pytest.approx(float(-logp.mean()))
    assert r["value"] == pytest.approx(float(np.exp(-logp.mean())))
    assert r["n"] == 100


def test_pplxm_rises_as_the_model_gets_worse():
    """More negative log-probs mean higher perplexity."""
    vals = [pp(np.full(20, -c))["value"] for c in (0.5, 1.0, 2.0, 4.0)]
    assert vals == sorted(vals)


def test_pplxm_is_the_geometric_not_the_arithmetic_mean():
    """exp(mean(nll)) != mean(exp(nll)) whenever the losses differ.

    Averaging on the wrong scale is the classic perplexity mistake, and it
    always over-states: here 7.39 vs 27.8.
    """
    logp = np.array([0.0, -4.0])
    assert pp(logp)["value"] == pytest.approx(np.exp(2.0))
    assert pp(logp)["value"] != pytest.approx(float(np.mean(np.exp(-logp))))


def test_pplxm_base_two_input_is_converted_to_nats():
    """log2 p = -3 is p = 1/8, so a uniform-over-8 model gives PPL = 8."""
    assert pp(np.full(100, -3.0), base="2")["value"] == pytest.approx(8.0)
    # And the same distribution expressed in nats must agree.
    assert pp(np.full(100, np.log(1 / 8)), base="e")["value"] == pytest.approx(8.0)


def test_pplxm_rejects_an_unknown_base_and_an_empty_sequence():
    with pytest.raises(ValueError, match="base must be"):
        pp(np.zeros(5), base="10")
    with pytest.raises(ValueError, match="at least one"):
        pp(np.array([]))
