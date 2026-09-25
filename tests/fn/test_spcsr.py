"""Tests for spcsr.schabenberger_csr_def."""

import math

import pytest

from morie.fn.spcsr import schabenberger_csr_def


def test_spcsr_basic():
    """A lattice of spacing 0.125 in a 1 x 1.25 region: lambda = 64, the CSR
    nearest-neighbour mean is 1/(2 sqrt 64) = 0.0625, every event's
    nearest neighbour is 0.125 away, so Clark-Evans is exactly 2."""
    pts = [[0.0625 + 0.125 * i, 0.0625 + 0.125 * j] for i in range(8) for j in range(10)]
    r = schabenberger_csr_def(pts, (0.0, 0.0, 1.0, 1.25))
    assert r["lambda_est"] == pytest.approx(64.0, rel=1e-15)
    assert r["expected_nn"] == pytest.approx(0.0625, rel=1e-15)
    assert r["mean_nn"] == pytest.approx(0.125, rel=1e-12)
    assert r["clark_evans"] == pytest.approx(2.0, rel=1e-12)
    c = [float(v) for v in r["quadrat_counts"].tolist()]
    assert r["n_quadrats"] == 16 and sum(c) == 80.0


def test_spcsr_edge():
    """Clustering pushes the index of dispersion above 1 and Clark-Evans
    below 1; the index is the sample variance over the mean of the counts."""
    pts = []
    for cx, cy in ((0.2, 0.2), (0.8, 0.3), (0.5, 0.85)):
        for k in range(20):
            a = 2 * math.pi * k / 20
            pts.append([cx + 0.03 * math.cos(a) * (1 + k % 3), cy + 0.03 * math.sin(a) * (1 + k % 3)])
    r = schabenberger_csr_def(pts, (0.0, 0.0, 1.0, 1.0))
    c = [float(v) for v in r["quadrat_counts"].tolist()]
    m = sum(c) / len(c)
    assert r["index_of_dispersion"] == pytest.approx(
        sum((v - m) ** 2 for v in c) / (len(c) - 1) / m, rel=1e-12)
    assert r["index_of_dispersion"] > 1 and r["clark_evans"] < 1


