"""Tests for clstcr.cluster_causal_inference."""

from morie.fn import _array_core as np
import pytest

from morie.fn.clstcr import cluster_causal_inference


def test_clstcr_basic():
    rng = np.random.default_rng(42)
    G, npc = 40, 50
    n = G * npc
    cl = np.repeat(np.arange(G), npc)
    D = np.repeat((rng.random(G) < 0.5).astype(float), npc)
    y = 1.0 * D + np.repeat(rng.normal(scale=1.5, size=G), npc) + rng.normal(scale=0.5, size=n)
    out = cluster_causal_inference(y, D, cl)
    assert out["se_cluster"] > 3 * out["se_naive"]
    assert out["n_clusters"] == G
    assert out["icc"] > 0.5


def test_clstcr_edge():
    with pytest.raises(ValueError):
        cluster_causal_inference(np.zeros(10), np.zeros(10), np.zeros(10))  # 1 cluster
    with pytest.raises(ValueError):
        cluster_causal_inference(np.zeros(9), np.full(9, 0.5), np.arange(9))  # non-binary D
