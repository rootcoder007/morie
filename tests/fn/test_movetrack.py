import math

from morie.fn.movetrack import (
    core_periphery,
    crw_kalman,
    movement_hmm,
    movement_network,
    network_robustness,
    track_steps,
)


def test_track_steps_ltraj_example():
    r = track_steps([0, 1, 2, 2, 3], [0, 0, 1, 2, 2], [0, 60, 120, 180, 240])
    assert r.dt[:4] == [60.0] * 4 and r.R2n == [0.0, 1.0, 5.0, 8.0, 13.0]
    assert abs(r.rel_angle[1] - math.pi / 4) < 1e-15 and abs(r.rel_angle[3] + math.pi / 2) < 1e-15
    assert r.rel_angle[0] is None and r.abs_angle[4] is None


def test_movement_network_counts():
    x = [0.5, 1.5, 1.6, 0.4, 1.5, 2.5]
    y = [0.5, 0.5, 0.4, 0.6, 0.5, 0.5]
    r = movement_network(x, y, 1.0)
    assert r.nodes == [(0, 0), (1, 0), (2, 0)] and r.edges == [(0, 1, 2), (1, 0, 1), (1, 2, 1)]
    assert r.visits == [2, 3, 1] and r.strength == [3, 4, 1]


def test_core_periphery_stationarity():
    A = [[0, 1, 1, 1, 1], [1, 0, 1, 0, 0], [1, 1, 0, 1, 0], [1, 0, 1, 0, 0], [1, 0, 0, 0, 0]]
    c = core_periphery(A).coreness
    n = 5
    sq = sum(v * v for v in c)
    for i in range(n):
        assert abs(c[i] - sum(A[i][j] * c[j] for j in range(n) if j != i) / (sq - c[i] ** 2)) < 1e-10


def test_robustness_complete_graph():
    n = 6
    K = [[0 if i == j else 1 for j in range(n)] for i in range(n)]
    r = network_robustness(K)
    assert abs(r.R - sum((n - q) / n for q in range(1, n + 1)) / n) < 1e-15


def test_hmm_likelihood_increases_and_decodes():
    st = [0.2, 0.3, 0.25, 0.2, 2.5, 3.0, 2.8, 2.6, 0.3, 0.2, 0.25, 3.1, 2.9, 0.22, 0.28]
    an = [None, 2.5, -2.8, 3.0, 0.1, -0.1, 0.05, 0.0, 2.9, -3.0, 2.7, 0.1, -0.05, 2.8, -2.9]
    r = movement_hmm(st, an, shape=[2, 2], scale=[0.1, 1.5], mu=[3.1, 0], kappa=[1, 1])
    assert all(b >= a - 1e-9 for a, b in zip(r.loglik, r.loglik[1:]))
    assert r.states == [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0]
    assert all(abs(sum(row) - 1) < 1e-12 for row in r.transition)


def test_crw_kalman_is_ml():
    xs = [0.0, 1.1, 2.3, 3.2, 4.4, 5.3, 6.6, 7.4, 8.1, 9.3, 10.0, 10.9]
    ys = [0.0, 0.2, 0.1, 0.5, 0.4, 0.9, 1.0, 1.3, 1.1, 1.6, 1.4, 1.9]
    r = crw_kalman(xs, ys)
    for d in ((0.01, 1, 1), (-0.01, 1, 1), (0, 1.02, 1), (0, 0.98, 1), (0, 1, 1.05), (0, 1, 0.95)):
        g = min(max(r.gamma + d[0], 1e-6), 0.999)
        other = crw_kalman(xs, ys, gamma=g, sigma=r.sigma * d[1], tau=r.tau * d[2]).loglik
        assert other <= r.loglik + 1e-6
