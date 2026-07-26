"""topkd: top-k decoding (Fan et al. 2018)."""

import numpy as np
import pytest

from morie.fn.topkd import top_k_decoding as tk


def test_topkd_keeps_exactly_k_candidates():
    rng = np.random.default_rng(2101)
    r = tk(rng.standard_normal(50), k=7)
    assert np.asarray(r["topk_indices"]).size == 7
    assert r["k"] == 7


def test_topkd_selects_the_largest_logits():
    z = np.array([0.1, 5.0, 0.2, 3.0, 0.3])
    idx = sorted(np.asarray(tk(z, k=2)["topk_indices"]).tolist())
    assert idx == [1, 3]


def test_topkd_output_is_a_distribution_over_the_kept_set():
    rng = np.random.default_rng(2111)
    r = tk(rng.standard_normal(40), k=5)
    p = np.asarray(r["tensor"])
    assert p.sum() == pytest.approx(1.0)
    assert np.count_nonzero(p) == 5


def test_topkd_k_equal_to_vocab_is_plain_softmax():
    z = np.array([1.0, 2.0, 3.0])
    e = np.exp(z - z.max())
    assert np.asarray(tk(z, k=3)["tensor"]) == pytest.approx(e / e.sum())


def test_topkd_k_one_is_greedy_decoding():
    z = np.array([0.5, 9.0, 2.0])
    p = np.asarray(tk(z, k=1)["tensor"])
    assert p[1] == pytest.approx(1.0)
    assert p[0] == 0.0 and p[2] == 0.0


def test_topkd_preserves_relative_odds_among_survivors():
    """Truncation renormalises; it must not reweight."""
    z = np.log(np.array([0.4, 0.3, 0.2, 0.1]))
    p = np.asarray(tk(z, k=2)["tensor"])
    assert p[0] / p[1] == pytest.approx(0.4 / 0.3)


def test_topkd_temperature_interacts_as_expected():
    """Lower temperature sharpens within the surviving set."""
    z = np.array([3.0, 2.0, 1.0, 0.0])
    hot = np.asarray(tk(z, k=3, T=5.0)["tensor"])
    cold = np.asarray(tk(z, k=3, T=0.2)["tensor"])
    assert cold.max() > hot.max()
