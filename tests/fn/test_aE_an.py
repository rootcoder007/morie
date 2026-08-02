"""Tests for aE_an.autoencoder_anomaly (linear-bottleneck anomaly score)."""

import numpy as np

from morie.fn.aE_an import autoencoder_anomaly


def _data_with_outlier():
    rng = np.random.default_rng(42)
    # rank-2 structure in 5-d: X = U @ V + small noise
    u = rng.normal(0, 1, (60, 2))
    v = rng.normal(0, 1, (2, 5))
    x = u @ v + rng.normal(0, 0.05, (60, 5))
    x[7] = [8.0, -8.0, 8.0, -8.0, 8.0]  # far off the rank-2 subspace
    return x


def test_aE_an_flags_planted_outlier():
    x = _data_with_outlier()
    result = autoencoder_anomaly(x, k=2, seed=0)
    scores = list(result["score"])
    assert int(max(range(len(scores)), key=lambda i: scores[i])) == 7
    assert bool(result["anomaly"][7])


def test_aE_an_rank_and_explained_fraction():
    x = _data_with_outlier()
    result = autoencoder_anomaly(x, k=2, seed=0)
    ranks = [int(v) for v in result["rank"]]
    assert sorted(ranks) == list(range(60))   # a permutation of 0..n-1
    assert ranks[7] == 0                      # planted outlier ranked first
    # rank-2 data (plus tiny noise): bottleneck of 2 explains almost all
    assert result["explained_fraction"] > 0.8
    assert len(result["reconstruction"]) == 60
