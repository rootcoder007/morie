"""Tests for Monte Carlo spatial test."""

from morie.fn import _array_core as np
from morie.fn.sgmci import sgmci


def test_sgmci_smoke():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 10, (30, 2))
    Z = rng.normal(0, 1, 30)
    r = sgmci(Z, coords, n_sim=49)
    assert r.name == "monte_carlo_spatial_test"
    assert "envelope_lo" in r.extra
    assert "envelope_hi" in r.extra


def test_cheatsheet():
    from morie.fn.sgmci import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0


def _moran(z, c):
    import math

    n = len(z)
    m = sum(z) / n
    d = [v - m for v in z]
    s0 = num = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                w = 1.0 / math.dist(c[i], c[j])
                s0 += w
                num += w * d[i] * d[j]
    return n / s0 * num / sum(v * v for v in d)


C = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.2], [0.3, 2.0], [1.1, 2.2], [2.4, 2.1]]
Z = [1.0, 1.2, 0.9, 1.1, 2.0, 2.4, 2.2, 3.1, 3.3]


def test_default_statistic_is_inverse_distance_moran():
    import pytest

    r = sgmci(Z, C, n_sim=39)
    assert r.value == pytest.approx(_moran(Z, C), rel=1e-13)


def test_permutations_are_philox_fisher_yates():
    """Recompute the permutation distribution from the documented stream."""
    import math

    import pytest

    from morie.fn._rng import random_uniform

    n, B = len(Z), 19
    u = random_uniform(B * (n - 1), seed=7, stream=0)
    sims, pos = [], 0
    for _ in range(B):
        p = list(Z)
        for i in range(n - 1, 0, -1):
            j = int(math.floor(float(u[pos]) * (i + 1)))
            pos += 1
            p[i], p[j] = p[j], p[i]
        sims.append(_moran(p, C))
    obs = _moran(Z, C)
    r = sgmci(Z, C, n_sim=B, seed=7)
    assert r.extra["p_value"] == pytest.approx((1 + sum(t >= obs for t in sims)) / (B + 1), rel=1e-15)
    assert r.extra["sim_mean"] == pytest.approx(sum(sims) / B, rel=1e-12)


def test_location_blind_statistic_cannot_reject():
    """A statistic that ignores the locations (here the maximum) is
    permutation-invariant: every replicate ties the observed value."""
    r = sgmci(Z, C, stat_fn=lambda z, c: max(z), n_sim=19)
    assert r.extra["p_value"] == 1.0
