"""Tests for csrkstst.kstest_csr.

The generated placeholders only asserted that a dict came back, and they
passed a 100-length normal vector as the observation window. They are
replaced by tests of what the procedure must actually do: hold its size
under the null, and have power against clustered and regular patterns.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.csrkstst import kstest_csr


def _csr(n=120, seed=0):
    return np.random.default_rng(seed).uniform(0, 1, (n, 2))


def _clustered(n=120, seed=0, k=6, sd=0.02):
    """Points tightly packed around k parents -- G rises far too early."""
    rng = np.random.default_rng(seed)
    parents = rng.uniform(0.1, 0.9, (k, 2))
    pts = parents[rng.integers(0, k, n)] + rng.normal(0, sd, (n, 2))
    return np.clip(pts, 0, 1)


def _regular(n=100, jitter=0.005, seed=0):
    """A jittered grid -- neighbours are all at nearly the same distance."""
    side = int(np.sqrt(n))
    g = np.linspace(0.05, 0.95, side)
    pts = np.array([[x, y] for x in g for y in g])
    return pts + np.random.default_rng(seed).normal(0, jitter, pts.shape)


def test_csr_pattern_is_not_rejected():
    """A pattern drawn from the null must not be flagged."""
    res = kstest_csr(_csr(seed=1), nsim=99, seed=7)
    assert res["p_value"] > 0.05


def test_clustered_pattern_is_rejected():
    res = kstest_csr(_clustered(seed=2), nsim=99, seed=7)
    assert res["p_value"] <= 0.05


def test_regular_pattern_is_rejected():
    res = kstest_csr(_regular(seed=3), nsim=99, seed=7)
    assert res["p_value"] <= 0.05


def test_clustering_shortens_nearest_neighbour_distance():
    """The direction the book states: clustered patterns give smaller mean NN."""
    assert kstest_csr(_clustered(seed=4), nsim=9, seed=7)["mean_nn"] < kstest_csr(_csr(seed=4), nsim=9, seed=7)["mean_nn"]


def test_p_value_is_a_rank_and_respects_its_bounds():
    """p = (1 + #{D_i >= D_obs}) / (1 + nsim), so it can never be 0."""
    res = kstest_csr(_clustered(seed=5), nsim=99, seed=7)
    assert res["p_value"] >= 1 / 100
    assert res["p_value"] <= 1.0
    assert np.isclose(res["p_value"] * 100 % 1, 0), "p must be a multiple of 1/(1+nsim)"


def test_seed_makes_the_p_value_reproducible():
    a = kstest_csr(_csr(seed=6), nsim=49, seed=99)["p_value"]
    b = kstest_csr(_csr(seed=6), nsim=49, seed=99)["p_value"]
    assert a == b


def test_supplied_cdf_skips_the_simulation():
    from scipy import stats

    res = kstest_csr(_csr(seed=8), cdf=stats.uniform(0, 0.2).cdf)
    assert res["nsim"] == 0
    assert 0.0 <= res["p_value"] <= 1.0


def test_nearest_neighbour_distances_are_correct():
    """Three points on a line: NN distances are 1, 1, 2."""
    res = kstest_csr(np.array([[0.0], [1.0], [3.0]]), nsim=5, seed=1)
    assert sorted(res["nn_distances"]) == [1.0, 1.0, 2.0]


def test_window_is_read_as_per_dimension_ranges():
    P = _csr(seed=9)
    for w in ([[0, 1], [0, 1]], [0, 1, 0, 1], np.array([[0, 1], [0, 1]])):
        assert 0 < kstest_csr(P, window=w, nsim=9, seed=1)["p_value"] <= 1


def test_validates_inputs():
    with pytest.raises(ValueError, match="at least 3 events"):
        kstest_csr(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="must be finite"):
        kstest_csr(np.array([[0.0, 0.0], [1.0, np.nan], [2.0, 1.0]]))
    with pytest.raises(ValueError, match="upper bounds must exceed"):
        kstest_csr(_csr(seed=9), window=[1, 0, 1, 0], nsim=5)
    with pytest.raises(ValueError, match="nsim must be at least 1"):
        kstest_csr(_csr(seed=9), nsim=0)
