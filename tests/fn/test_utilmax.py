import itertools

from morie.fn.utilmax import quadratic_utility_constrained, quadratic_utility_lagrange, vcg_mechanism


def test_lagrange_stationarity():
    W = [[2.0, 0.5], [0.5, 1.0]]
    xs = [1.5, -0.5]
    r = quadratic_utility_lagrange(xs, W, [[1.0, 2.0]], [0.3])
    assert abs(r.x[0] + 2 * r.x[1] - 0.3) < 1e-12
    g = [2 * sum(W[i][j] * (xs[j] - r.x[j]) for j in range(2)) for i in range(2)]
    assert abs(g[0] - r.multipliers[0] * 1.0) < 1e-12 and abs(g[1] - r.multipliers[0] * 2.0) < 1e-12


def test_constrained_beats_every_grid_point():
    W = [[1.0, 0.3], [0.3, 2.0]]
    xs = [3.0, 2.0]
    A, b = [[1, 1], [-1, 0], [0, -1]], [2.0, 0.0, 0.0]
    r = quadratic_utility_constrained(xs, W, A, b)

    def u(x):
        return -sum((x[i] - xs[i]) * W[i][j] * (x[j] - xs[j]) for i in range(2) for j in range(2))

    for i, j in itertools.product(range(41), repeat=2):
        x = (2 * i / 40, 2 * j / 40)
        if x[0] + x[1] <= 2 + 1e-12:
            assert u(x) <= r.utility + 1e-12
    assert r.active == [0]


def test_vcg_clarke_payments():
    V = [[5, 0, 1], [0, 3, 2], [0, 4, 2], [1, 1, 1]]
    r = vcg_mechanism(V)
    tot = [sum(V[i][k] for i in range(4)) for k in range(3)]
    k = tot.index(max(tot))
    assert r.choice == k
    for i in range(4):
        others = [tot[c] - V[i][c] for c in range(3)]
        assert r.payments[i] == max(others) - others[k] >= 0
