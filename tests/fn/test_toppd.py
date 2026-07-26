"""toppd: nucleus (top-p) sampling (Holtzman et al. 2020)."""

import numpy as np
import pytest

from morie.fn.toppd import top_p_nucleus as tp


def test_toppd_p_equal_one_keeps_everything():
    rng = np.random.default_rng(1801)
    r = tp(rng.standard_normal(20), p=1.0)
    assert bool(np.all(np.asarray(r["keep_mask"])))
    assert r["n_kept"] == 20


def test_toppd_output_is_a_distribution_over_the_kept_set():
    rng = np.random.default_rng(1811)
    r = tp(rng.standard_normal(30), p=0.9)
    probs = np.asarray(r["tensor"])
    keep = np.asarray(r["keep_mask"])
    assert probs.sum() == pytest.approx(1.0)
    assert np.all(probs[~keep] == 0.0), "discarded tokens must have zero mass"


def test_toppd_keeps_the_smallest_set_whose_mass_reaches_p():
    """The nucleus is defined by cumulative mass, not by a count."""
    z = np.log(np.array([0.5, 0.3, 0.15, 0.05]))
    r = tp(z, p=0.75)
    # 0.5 alone is below 0.75; 0.5+0.3 = 0.8 reaches it, so 2 tokens.
    assert r["n_kept"] == 2
    keep = np.asarray(r["keep_mask"])
    assert keep[0] and keep[1] and not keep[2] and not keep[3]


def test_toppd_a_tiny_p_still_keeps_at_least_the_argmax():
    """Keeping nothing would make sampling impossible."""
    r = tp(np.log(np.array([0.6, 0.3, 0.1])), p=1e-9)
    assert r["n_kept"] >= 1
    assert np.asarray(r["keep_mask"])[0]


def test_toppd_shrinks_monotonically_as_p_falls():
    rng = np.random.default_rng(1823)
    z = rng.standard_normal(50)
    kept = [tp(z, p=q)["n_kept"] for q in (1.0, 0.9, 0.7, 0.5, 0.2)]
    assert kept == sorted(kept, reverse=True)


def test_toppd_preserves_the_relative_odds_of_kept_tokens():
    """Nucleus sampling renormalises; it does not reweight."""
    z = np.log(np.array([0.4, 0.3, 0.2, 0.1]))
    r = tp(z, p=0.7)
    probs = np.asarray(r["tensor"])
    assert probs[0] / probs[1] == pytest.approx(0.4 / 0.3)


def test_toppd_is_invariant_to_a_constant_logit_shift():
    z = np.array([2.0, 1.0, 0.5, -3.0])
    assert np.asarray(tp(z + 9.0, p=0.8)["tensor"]) == pytest.approx(
        np.asarray(tp(z, p=0.8)["tensor"])
    )
