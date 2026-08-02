"""Tests for kmnsc.kmeans_clustering."""

from morie.fn import _array_core as np
import pytest

from morie.fn.kmnsc import kmeans_clustering


def _blobs(seed=0, n_per=50, spread=0.3):
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [6.0, 0.0], [0.0, 6.0]])
    X = np.vstack([c + rng.normal(0, spread, size=(n_per, 2)) for c in centers])
    return X, centers


def test_kmnsc_k_equals_one_inertia_is_the_total_sum_of_squares():
    """With one cluster the centre is the mean and the inertia is exactly
    sum ||x - xbar||^2 -- an identity, not an approximation."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 3))
    r = kmeans_clustering(X, n_clusters=1)
    tss = float(((X - X.mean(axis=0)) ** 2).sum())
    assert float(r["inertia"]) == pytest.approx(tss, rel=1e-9)
    np.testing.assert_allclose(np.asarray(r["centers"])[0], X.mean(axis=0), atol=1e-9)


def test_kmnsc_recovers_well_separated_blobs():
    X, centers = _blobs()
    r = kmeans_clustering(X, n_clusters=3, seed=0)
    found = np.asarray(r["centers"], dtype=float)
    # Each true centre has a found centre within a fraction of the spread.
    dists = np.linalg.norm(found[:, None, :] - centers[None, :, :], axis=2)
    assert float(dists.min(axis=0).max()) < 0.2
    # Each blob lands in exactly one cluster.
    labels = np.asarray(r["labels"])
    for b in range(3):
        blob = labels[b * 50 : (b + 1) * 50]
        assert len(set(blob.tolist())) == 1


def test_kmnsc_inertia_never_increases_with_k():
    """More clusters can only lower the within-cluster sum of squares."""
    X, _ = _blobs(seed=2)
    inert = [float(kmeans_clustering(X, n_clusters=k, seed=0)["inertia"]) for k in (1, 2, 3, 4)]
    assert inert[0] > inert[1] > inert[2] >= inert[3]
    # The elbow: going 2 -> 3 on three blobs is a huge drop, 3 -> 4 is not.
    assert (inert[1] - inert[2]) > 10 * (inert[2] - inert[3])
