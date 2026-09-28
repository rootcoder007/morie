from morie.fn._rng import random_uniform
from morie.fn.communities import fast_greedy_modularity, graph_modularity, walktrap_communities


def _graph():
    u = [float(v) for v in random_uniform(2000, seed=9)]
    n, t = 24, 0
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if u[t] < (0.55 if i // 8 == j // 8 else 0.06):
                A[i][j] = A[j][i] = round(0.5 + 2 * u[t + 1], 3)
            t += 2
    return A


def _modularity(A, lab):  # Newman: (1/2m) sum_ij (A_ij - k_i k_j / 2m) delta(c_i, c_j)
    n = len(A)
    k = [sum(r) for r in A]
    m2 = sum(k)
    return sum(A[i][j] - k[i] * k[j] / m2 for i in range(n) for j in range(n) if lab[i] == lab[j]) / m2


def test_modularity_definition():
    A = _graph()
    lab = [i // 8 for i in range(24)]
    assert abs(graph_modularity(A, lab) - _modularity(A, lab)) < 1e-12


def test_fast_greedy_and_walktrap_report_their_modularity():
    A = _graph()
    for r in (fast_greedy_modularity(A), walktrap_communities(A)):
        assert abs(r.max_modularity - _modularity(A, r.membership)) < 1e-12
        assert r.max_modularity == max(r.modularity) and len(r.modularity) == len(r.merges) + 1
    w = walktrap_communities(A)
    assert w.membership == [1, 1, 2, 1, 1, 1, 1, 1] + [2] * 8 + [3] * 8
