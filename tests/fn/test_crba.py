"""Tests for crba.crba (Cronbach's alpha)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.crba import crba


def test_crba_matches_a_hand_computed_alpha():
    """Two items with item 2 = 2 x item 1: item variances (1, 4), total
    variance 9, so raw alpha = 2(1 - 5/9) = 8/9. Perfect inter-item
    correlation makes the standardised alpha exactly 1."""
    X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    r = crba(X)
    assert r.raw == pytest.approx(8 / 9, rel=1e-12)
    assert r.std == pytest.approx(1.0, rel=1e-12)
    assert r.avgr == pytest.approx(1.0, rel=1e-12)
    assert r.k == 2 and r.n == 3


def test_crba_grows_with_test_length_like_spearman_brown():
    """Adding parallel items raises alpha: with common variance and
    inter-item correlation rho, alpha = k rho / (1 + (k-1) rho). Doubling
    k from 4 to 8 at the same rho must increase the standardised alpha."""
    rng = np.random.default_rng(0)
    n = 500
    true = rng.normal(size=n)
    items8 = np.column_stack([true + rng.normal(0, 1.0, n) for _ in range(8)])
    a4 = crba(items8[:, :4]).std
    a8 = crba(items8).std
    assert a8 > a4
    # rho ~ 0.5 here, so Spearman-Brown predicts ~0.80 at k=4, ~0.89 at k=8.
    assert a4 == pytest.approx(4 * 0.5 / (1 + 3 * 0.5), abs=0.08)
    assert a8 == pytest.approx(8 * 0.5 / (1 + 7 * 0.5), abs=0.05)


def test_crba_uncorrelated_items_give_alpha_near_zero():
    rng = np.random.default_rng(1)
    r = crba(rng.normal(size=(400, 5)))
    assert abs(r.raw) < 0.15
    assert r.ci_lo < r.raw < r.ci_hi


def test_crba_single_item_returns_nan():
    r = crba(np.arange(10.0).reshape(-1, 1))
    assert np.isnan(r.raw) and np.isnan(r.std)
