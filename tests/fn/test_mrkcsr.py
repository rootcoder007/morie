"""Tests for mrkcsr.csr_test (Ripley's K with Monte Carlo envelopes)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.mrkcsr import _ripley_k, csr_test


def _csr(n=120, seed=0):
    return np.random.default_rng(seed).uniform(0, 1, (n, 2))


def _clustered(n=120, seed=0, k=6, sd=0.02):
    rng = np.random.default_rng(seed)
    parents = rng.uniform(0.1, 0.9, (k, 2))
    return np.clip(parents[rng.integers(0, k, n)] + rng.normal(0, sd, (n, 2)), 0, 1)


def test_k_is_non_decreasing_in_r():
    """K(r) counts pairs within r, so it cannot fall as r grows."""
    P = _csr(seed=1)
    r = np.linspace(0.02, 0.2, 10)
    k = _ripley_k(P, r, 1.0)
    assert np.all(np.diff(k) >= -1e-12)


def test_k_approximates_pi_r_squared_for_csr_away_from_the_edge():
    """Uncorrected K is biased low near the boundary, so this is checked
    at a small radius on a large sample where the bias is slight."""
    P = _csr(n=3000, seed=2)
    r = np.array([0.03])
    assert _ripley_k(P, r, 1.0)[0] == pytest.approx(np.pi * 0.03**2, rel=0.15)


def test_csr_pattern_is_not_rejected():
    assert csr_test(_csr(seed=3), nsim=99, seed=7)["p_value"] > 0.05


def test_clustered_pattern_is_rejected():
    assert csr_test(_clustered(seed=4), nsim=99, seed=7)["p_value"] <= 0.05


def test_clustered_k_exceeds_the_simulated_mean():
    """Clustering puts more pairs at short range than CSR does."""
    res = csr_test(_clustered(seed=5), nsim=49, seed=7)
    assert res["k_observed"][0] > res["k_mean"][0]


def test_envelopes_bracket_the_mean():
    res = csr_test(_csr(seed=6), nsim=49, seed=7)
    assert np.all(res["k_lower"] <= res["k_mean"] + 1e-12)
    assert np.all(res["k_upper"] >= res["k_mean"] - 1e-12)


def test_p_value_is_a_rank():
    res = csr_test(_clustered(seed=8), nsim=99, seed=7)
    assert res["p_value"] >= 1 / 100
    assert np.isclose(res["p_value"] * 100 % 1, 0)


def test_seed_makes_it_reproducible():
    P = _csr(seed=9)
    assert csr_test(P, nsim=29, seed=4)["p_value"] == csr_test(P, nsim=29, seed=4)["p_value"]


def test_validates_inputs():
    P = _csr(seed=10)
    with pytest.raises(ValueError, match="at least 3 events"):
        csr_test(P[:2])
    with pytest.raises(ValueError, match="must be finite"):
        bad = P.copy(); bad[0, 0] = np.inf
        csr_test(bad)
    with pytest.raises(ValueError, match="upper bounds must exceed"):
        csr_test(P, window=[1, 0, 1, 0])
    with pytest.raises(ValueError, match="nsim must be at least 1"):
        csr_test(P, nsim=0)
    with pytest.raises(ValueError, match="radii must be positive"):
        csr_test(P, radii=[-1.0])
