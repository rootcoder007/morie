"""Tests for graphlogic: graph indices, CDCL satisfiability and linear-chain CRFs."""

import itertools
import math

from morie.fn.graphlogic import (
    cdcl_solve,
    crf_fit,
    crf_marginals,
    crf_viterbi,
    graph_betweenness,
    graph_degree_centrality,
    graph_diameter,
)

N = 10
A = [[1 if (i != j and ((i * 7 + j * 3) % 5 == 0 or abs(i - j) == 1)) else 0 for j in range(N)] for i in range(N)]
A = [[max(A[i][j], A[j][i]) for j in range(N)] for i in range(N)]


def _floyd(M, weighted=False):
    n = len(M)
    d = [
        [0.0 if i == j else (M[i][j] if weighted else 1.0) if M[i][j] else math.inf for j in range(n)] for i in range(n)
    ]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if d[i][k] + d[k][j] < d[i][j]:
                    d[i][j] = d[i][k] + d[k][j]
    return d


def test_degree_centrality():
    assert graph_degree_centrality(A) == [sum(1 for j in range(N) if A[i][j]) / (N - 1) for i in range(N)]
    D = [[0, 1, 1], [0, 0, 1], [0, 0, 0]]
    assert graph_degree_centrality(D, mode="out") == [1.0, 0.5, 0.0]
    assert graph_degree_centrality(D, mode="in") == [0.0, 0.5, 1.0]
    assert graph_degree_centrality(D) == [1.0, 1.0, 1.0]


def test_diameter_matches_floyd_warshall():
    W = [[(1 + (i * j) % 4) if A[i][j] else 0 for j in range(N)] for i in range(N)]
    for M, w in ((A, False), (W, True)):
        d = _floyd(M, w)
        r = graph_diameter(M, weighted=w)
        assert r.eccentricity == [max(row) for row in d]
        assert r.diameter == max(max(row) for row in d)
        assert d[r.pair[0]][r.pair[1]] == r.diameter


def test_betweenness_matches_path_counting():
    d = _floyd(A)
    # count shortest paths by dynamic programming on the distance layers
    sig = [[0.0] * N for _ in range(N)]
    for s in range(N):
        sig[s][s] = 1.0
        for t in sorted(range(N), key=lambda v: d[s][v]):
            if t != s:
                sig[s][t] = sum(sig[s][u] for u in range(N) if A[u][t] and d[s][u] + 1 == d[s][t])
    want = [0.0] * N
    for v in range(N):
        for s in range(N):
            for t in range(s + 1, N):
                if v not in (s, t) and d[s][v] + d[v][t] == d[s][t]:
                    want[v] += sig[s][v] * sig[v][t] / sig[s][t]
    got = graph_betweenness(A)
    assert max(abs(a - b) for a, b in zip(got, want)) <= 1e-12
    norm = graph_betweenness(A, normalized=True)
    assert abs(norm[3] - got[3] / ((N - 1) * (N - 2) / 2)) <= 1e-15


def _sat(cnf, model):
    s = set(model)
    return all(any(x in s for x in c) for c in cnf)


def test_cdcl_models_and_brute_force_agreement():
    for seed in range(12):
        cnf = [
            [((k * 7 + j * 3 + seed * 5) % 7 + 1) * (1 if (k + j + seed) % 3 else -1) for j in range(3)]
            for k in range(30 + seed)
        ]
        r = cdcl_solve(cnf)
        brute = any(
            _sat(cnf, [v if b else -v for v, b in zip(range(1, 8), bits)])
            for bits in itertools.product([0, 1], repeat=7)
        )
        assert r.satisfiable == brute
        if r.satisfiable:
            assert _sat(cnf, r.model)
    php = [[1, 2], [3, 4], [5, 6], [-1, -3], [-1, -5], [-3, -5], [-2, -4], [-2, -6], [-4, -6]]
    assert cdcl_solve(php).satisfiable is False


SEQS = [[[math.sin(t + s), math.cos(0.7 * t * (s + 1))] for t in range(4 + s)] for s in range(3)]
LABS = [[(t + s) % 3 if math.sin(t + s) > -0.3 else 1 for t in range(4 + s)] for s in range(3)]


def _score(theta, X, y, K=3, d=2):
    W = [theta[k * d : (k + 1) * d] for k in range(K)]
    b = theta[K * d : K * d + K]
    T = [theta[K * d + K + i * K : K * d + K + (i + 1) * K] for i in range(K)]
    s = sum(sum(W[y[t]][j] * X[t][j] for j in range(d)) + b[y[t]] for t in range(len(X)))
    return s + sum(T[y[t - 1]][y[t]] for t in range(1, len(X)))


def test_crf_marginals_and_viterbi_match_enumeration():
    theta = [0.3 * math.sin(i + 1) for i in range(3 * 2 + 3 + 9)]
    X = SEQS[1]
    ys = list(itertools.product(range(3), repeat=len(X)))
    sc = [_score(theta, X, y) for y in ys]
    lz = math.log(sum(math.exp(v) for v in sc))
    m = crf_marginals(theta, X, 3)
    assert abs(m.log_z - lz) <= 1e-12
    for t in range(len(X)):
        for k in range(3):
            want = sum(math.exp(v - lz) for y, v in zip(ys, sc) if y[t] == k)
            assert abs(m.node[t][k] - want) <= 1e-12
    best = max(range(len(ys)), key=lambda i: sc[i])
    assert crf_viterbi(theta, X, 3).labels == list(ys[best])


def test_crf_fit_is_stationary():
    fit = crf_fit(SEQS, LABS, 3, l2=0.5)

    def obj(th):
        out = 0.25 * sum(v * v for v in th)
        for X, y in zip(SEQS, LABS):
            out += crf_marginals(th, X, 3).log_z - _score(th, X, y)
        return out

    assert abs(obj(fit.theta) - fit.objective) <= 1e-10
    for i in range(len(fit.theta)):
        e = 1e-5
        up = list(fit.theta)
        dn = list(fit.theta)
        up[i] += e
        dn[i] -= e
        assert abs((obj(up) - obj(dn)) / (2 * e)) <= 1e-5
