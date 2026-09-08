"""latnh: Latin hypercube sampling (McKay, Beckman & Conover 1979)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.latnh import latin_hypercube as lh


def test_latnh_every_margin_is_perfectly_stratified():
    """The defining property: project onto any one dimension and each of the
    N equal-width strata contains exactly one point. Plain random sampling
    gives that with vanishing probability.
    """
    N, d = 50, 4
    s = np.asarray(lh(N=N, d=d, seed=1)["sample"])
    for j in range(d):
        strata = np.floor(s[:, j] * N).astype(int)
        assert sorted(strata) == list(range(N))


def test_latnh_shape_and_support():
    N, d = 30, 3
    r = lh(N=N, d=d, seed=2)
    s = np.asarray(r["sample"])
    assert s.shape == (N, d)
    assert np.all((s >= 0) & (s <= 1))
    assert r["N"] == N and r["d"] == d


def test_latnh_margins_are_close_to_uniform_mean():
    """Stratification forces each margin's mean near 1/2 far more tightly
    than iid sampling would at the same N."""
    s = np.asarray(lh(N=200, d=5, seed=3)["sample"])
    assert np.allclose(s.mean(axis=0), 0.5, atol=0.02)


def test_latnh_beats_iid_on_margin_mean_error():
    """Compare like for like at the same N over repeated draws."""
    rng = np.random.default_rng(4)
    N = 40
    lhs_err, iid_err = [], []
    for k in range(60):
        lhs_err.append(abs(np.asarray(lh(N=N, d=1, seed=k)["sample"]).mean() - 0.5))
        iid_err.append(abs(rng.random(N).mean() - 0.5))
    assert np.mean(lhs_err) < np.mean(iid_err)


def test_latnh_is_reproducible_and_seed_sensitive():
    a = np.asarray(lh(N=20, d=2, seed=9)["sample"])
    b = np.asarray(lh(N=20, d=2, seed=9)["sample"])
    c = np.asarray(lh(N=20, d=2, seed=10)["sample"])
    assert a == pytest.approx(b, abs=0.0)
    assert not np.allclose(a, c)


def test_latnh_dimensions_are_independently_permuted():
    """If every column used the same permutation the sample would lie on a
    diagonal and be useless for anything but 1-D."""
    s = np.asarray(lh(N=100, d=2, seed=11)["sample"])
    assert abs(float(np.corrcoef(s[:, 0], s[:, 1])[0, 1])) < 0.35
