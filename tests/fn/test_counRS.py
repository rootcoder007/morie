"""Tests for counRS.counterfactual_rec."""

from morie.fn import _array_core as np
import pytest

from morie.fn.counRS import counterfactual_rec


def _logged(seed=42, n=4000, k=3):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, k, n)
    true_r = np.array([0.1, 0.5, 0.9])
    r = true_r[a] + rng.normal(scale=0.1, size=n)
    target = np.zeros((n, k))
    target[:, 2] = 1.0
    return a, r, np.full(n, 1 / k), target, np.tile(true_r, (n, 1))


def test_counRS_basic():
    a, r, p0, target, q = _logged()
    out = counterfactual_rec(a, r, p0, target)
    assert out["ips"] == pytest.approx(0.9, abs=0.05)
    assert out["snips"] == pytest.approx(0.9, abs=0.05)
    dr = counterfactual_rec(a, r, p0, target, reward_model=q)
    assert dr["dr"] == pytest.approx(0.9, abs=0.03)


def test_counRS_edge():
    a, r, p0, target, q = _logged()
    with pytest.raises(ValueError):
        counterfactual_rec(a, r, np.zeros_like(p0), target)  # zero logging probability
    assert counterfactual_rec(a, r, p0, target, clip=1.0)["n_clipped"] > 0
