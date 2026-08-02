"""Tests for tsnrd.tsne_reduction."""

from morie.fn import _array_core as np
import pytest

from morie.fn.tsnrd import tsne_reduction


def _blobs(seed=0, n_per=30):
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    X = np.vstack([c + rng.normal(0, 0.5, size=(n_per, 3)) for c in centers])
    return X


def test_tsnrd_embeds_to_the_requested_dimension():
    X = _blobs()
    r = tsne_reduction(X, n_components=2, perplexity=10.0, seed=0)
    emb = np.asarray(r["embedding"], dtype=float)
    assert emb.shape == (90, 2)
    assert float(r["kl_divergence"]) >= 0


def test_tsnrd_separated_blobs_stay_separated():
    """Well-separated clusters in input space must remain separated in the
    embedding: mean between-cluster distance far above within-cluster."""
    X = _blobs(seed=1)
    emb = np.asarray(tsne_reduction(X, n_components=2, perplexity=10.0, seed=0)["embedding"], dtype=float)
    groups = [emb[i * 30 : (i + 1) * 30] for i in range(3)]
    within = np.mean([np.linalg.norm(g - g.mean(axis=0), axis=1).mean() for g in groups])
    centres = np.array([g.mean(axis=0) for g in groups])
    between = np.mean([np.linalg.norm(centres[i] - centres[j]) for i in range(3) for j in range(i + 1, 3)])
    assert between > 3 * within


def test_tsnrd_is_reproducible_with_a_seed():
    X = _blobs(seed=2)
    a = np.asarray(tsne_reduction(X, perplexity=10.0, seed=5)["embedding"], dtype=float)
    b = np.asarray(tsne_reduction(X, perplexity=10.0, seed=5)["embedding"], dtype=float)
    np.testing.assert_allclose(a, b, atol=1e-8)
