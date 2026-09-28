import bisect
import math

from morie.fn.clusterensemble import cluster_stability, consensus_clustering, evidence_accumulation


def _blobs():
    pts = []
    for c, (cx, cy) in enumerate([(0, 0), (6, 1), (2, 7)]):
        for i in range(15):
            a = (i * 2.399963 + c) % (2 * math.pi)
            r = 0.4 + 1.3 * ((i * 0.618034 + c * 0.3) % 1)
            pts.append([cx + r * math.cos(a), cy + r * math.sin(a)])
    return pts


TRUTH = [1] * 15 + [2] * 15 + [3] * 15


def test_consensus_matrix_and_summaries():
    r = consensus_clustering(_blobs(), 3, n_resamples=30, seed=2)
    assert r.cluster == TRUTH
    M = r.consensus
    assert all(M[i][j] == M[j][i] and 0 <= M[i][j] <= 1 for i in range(45) for j in range(45))
    vals = sorted(M[i][j] for i in range(45) for j in range(i + 1, 45))
    area = sum((vals[t] - vals[t - 1]) * bisect.bisect_right(vals, vals[t]) / len(vals) for t in range(1, len(vals)))
    assert abs(r.cdf_area - area) < 1e-12
    mem = [i for i in range(45) if r.cluster[i] == 2]
    pairs = [M[a][b] for x, a in enumerate(mem) for b in mem[x + 1 :]]
    assert abs(r.cluster_consensus[1] - sum(pairs) / len(pairs)) < 1e-12


def test_evidence_accumulation_lifetime():
    r = evidence_accumulation(_blobs(), n_runs=30, k_range=(5, 12))
    assert r.k == 3 and r.cluster == TRUTH
    C = r.coassociation
    assert all(abs(C[i][i] - 1) < 1e-12 for i in range(45))


def test_stability_jaccard_bounds():
    r = cluster_stability(_blobs(), 3, n_boot=20)
    for v, rec, js in zip(r.mean_jaccard, r.recovered, r.jaccard):
        assert all(0 <= x <= 1 for x in js)
        assert abs(v - sum(js) / len(js)) < 1e-12 and rec == sum(1 for x in js if x > 0.5)
