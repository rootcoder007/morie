"""Tests for hmddpm.geron_ddpm."""

import math

import pytest

from morie.fn.hmddpm import geron_ddpm


def test_hmddpm_basic():
    """Gradient descent on the per-timestep affine eps-model converges to
    the least-squares fit of eps_t on x_t, recomputed here from the same
    noise draws."""
    from morie.fn.hmdfw import lcg_normal
    X = [[1.0], [2.0], [3.0], [4.0], [2.5]]
    T, seed = 3, 2
    r = geron_ddpm(X, T=T, epochs=4000, lr=0.1, seed=seed)
    for t in range(T):
        ab = r["alpha_bar"][t]
        eps = [row[0] for row in lcg_normal((5, 1), seed + 1 + t).tolist()]
        xt = [math.sqrt(ab) * X[i][0] + math.sqrt(1 - ab) * eps[i] for i in range(5)]
        mx, me = sum(xt) / 5, sum(eps) / 5
        a = sum((x - mx) * (e - me) for x, e in zip(xt, eps)) / sum((x - mx) ** 2 for x in xt)
        assert r["A"][t] == pytest.approx(a, rel=1e-9)
        assert r["b"][t][0] == pytest.approx(me - a * mx, abs=1e-9)
    assert r["monotone"]


def test_hmddpm_edge():
    """alpha_bar is the running product of 1 - beta; bad inputs raise."""
    r = geron_ddpm([[0.5], [1.5]], T=5, epochs=5)
    prod = 1.0
    for b, ab in zip(r["betas"], r["alpha_bar"]):
        prod *= 1.0 - b
        assert ab == pytest.approx(prod, rel=1e-15)
    with pytest.raises(ValueError, match="T must be"):
        geron_ddpm([[1.0]], T=0)
    with pytest.raises(ValueError, match="lr must be"):
        geron_ddpm([[1.0]], lr=-1)


