"""Tests for morie.fn.zesir: conservation, isolation and one RK4 step recomputed."""

from morie.fn.zesir import spatial_sir

W = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
N = [1000.0, 800.0, 1200.0, 500.0]


def test_conservation_and_isolation():
    r = spatial_sir([5, 0, 0, 0], W, N, beta=0.5, gamma=0.2, t_max=40, dt=0.25)
    for s, i, q in zip(r.extra["S"], r.extra["I"], r.extra["R"]):
        for k in range(4):
            assert abs(s[k] + i[k] + q[k] - N[k]) < 1e-9
    iso = spatial_sir([5, 0, 0, 0], None, N, beta=0.5, gamma=0.2, t_max=40, dt=0.25)
    assert iso.extra["I"][-1][1] == 0.0 and iso.extra["R"][-1][2] == 0.0
    assert r.extra["R"][-1][3] > 0.0


def test_first_rk4_step():
    beta, gamma, kappa, dt = 0.5, 0.2, 0.1, 0.25
    Wt = [[v / sum(row) for v in row] for row in W]
    C = [[(1 - kappa) * (i == j) + kappa * Wt[i][j] for j in range(4)] for i in range(4)]

    def f(y):
        S, Inf = y[:4], y[4:8]
        lam = [sum(C[i][j] * Inf[j] / N[j] for j in range(4)) for i in range(4)]
        return (
            [-beta * S[i] * lam[i] for i in range(4)]
            + [beta * S[i] * lam[i] - gamma * Inf[i] for i in range(4)]
            + [gamma * Inf[i] for i in range(4)]
        )

    y0 = [N[0] - 5, N[1], N[2], N[3], 5, 0, 0, 0, 0, 0, 0, 0]
    k1 = f(y0)
    k2 = f([a + dt / 2 * b for a, b in zip(y0, k1)])
    k3 = f([a + dt / 2 * b for a, b in zip(y0, k2)])
    k4 = f([a + dt * b for a, b in zip(y0, k3)])
    y1 = [a + dt / 6 * (p + 2 * q + 2 * s + t) for a, p, q, s, t in zip(y0, k1, k2, k3, k4)]
    r = spatial_sir([5, 0, 0, 0], W, N, beta=beta, gamma=gamma, kappa=kappa, t_max=dt, dt=dt)
    got = r.extra["S"][1] + r.extra["I"][1] + r.extra["R"][1]
    assert max(abs(a - b) for a, b in zip(got, y1)) < 1e-10
