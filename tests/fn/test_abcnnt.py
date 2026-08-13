"""Tests for abcnnt (Papamakarios, Sterratt & Murray 2019, SNL)."""

import math

from morie.fn.abcnnt import (MAF, abcnnt, flow_forward, flow_logprob,
                             mcmc_sample, sequential_neural_likelihood,
                             train_flow)

SIG = 0.5
X_O = [1.2]


def _rng(seed):
    st = [seed]

    def uni():
        st[0] = (1103515245 * st[0] + 12345) % (1 << 31)
        return st[0] / float(1 << 31)

    def normal():
        return math.sqrt(-2 * math.log(max(uni(), 1e-12))) * \
            math.cos(2 * math.pi * uni())
    return uni, normal


def _sim(t, nrm):
    return [t[0] + SIG * nrm()]


def _log_prior(t):
    return -0.5 * t[0] * t[0]


def test_the_jacobian_is_triangular():
    flow = MAF(3, 2, n_layers=1, hidden=12, seed=4)
    L = flow["layers"][0]
    one = {"layers": [L], "dim_x": 3, "dim_t": 2}
    theta, base, h = [0.3, -0.7], [0.2, -0.5, 0.9], 1e-6
    J = [[0.0] * 3 for _ in range(3)]
    for j in range(3):
        up, dn = list(base), list(base)
        up[j] += h
        dn[j] -= h
        a = flow_forward(one, up, theta)[0]
        b = flow_forward(one, dn, theta)[0]
        for i in range(3):
            J[i][j] = (a[i] - b[i]) / (2 * h)
    order = L["order"]
    for i in range(3):
        for j in range(3):
            if i != j and order[j] >= order[i]:
                assert abs(J[i][j]) < 1e-9
    assert max(abs(J[i][j]) for i in range(3) for j in range(3)
               if order[j] < order[i]) > 1e-6


def test_log_det_matches_the_scales():
    flow = MAF(3, 2, n_layers=1, hidden=12, seed=4)
    L = flow["layers"][0]
    one = {"layers": [L], "dim_x": 3, "dim_t": 2}
    theta, base, h = [0.3, -0.7], [0.2, -0.5, 0.9], 1e-6
    det = 1.0
    for i in range(3):
        up, dn = list(base), list(base)
        up[i] += h
        dn[i] -= h
        det *= (flow_forward(one, up, theta)[0][i] -
                flow_forward(one, dn, theta)[0][i]) / (2 * h)
    _u, total = flow_forward(one, base, theta)
    assert abs(math.log(abs(det)) + total) < 1e-6


def test_the_density_integrates_to_one():
    f = MAF(1, 1, n_layers=3, hidden=10, seed=2)
    for cond in ([0.4], [-1.5]):
        tot, lo, hi, n = 0.0, -14.0, 14.0, 4000
        step = (hi - lo) / n
        for i in range(n):
            tot += math.exp(flow_logprob(f, [lo + (i + 0.5) * step],
                                         cond)) * step
        assert abs(tot - 1.0) < 3e-3


def test_the_conditioning_reaches_every_dimension():
    f = MAF(3, 2, n_layers=2, hidden=10, seed=4)
    a = flow_logprob(f, [0.2, -0.5, 0.9], [0.3, -0.7])
    b = flow_logprob(f, [0.2, -0.5, 0.9], [1.9, 2.2])
    assert abs(a - b) > 1e-6


def test_training_improves_and_learns_the_shape():
    uni, normal = _rng(77)
    D = []
    for _ in range(120):
        t = [-2.0 + 4.0 * uni()]
        D.append((t, [t[0] + SIG * normal()]))
    f = MAF(1, 1, n_layers=2, hidden=10, seed=1)
    before = sum(flow_logprob(f, x, t) for t, x in D) / len(D)
    train_flow(f, D, epochs=60, lr=0.02, seed=3)
    after = sum(flow_logprob(f, x, t) for t, x in D) / len(D)
    assert after > before
    grid = [-1.5 + 0.15 * k for k in range(21)]
    got = [flow_logprob(f, [0.0], [t]) for t in grid]
    peak = grid[max(range(len(grid)), key=lambda i: got[i])]
    assert abs(peak) < 0.5


def test_algorithm_1_on_a_conjugate_problem():
    res = abcnnt(_sim, X_O, _log_prior, [0.0], n_rounds=3,
                 n_per_round=40, n_layers=2, hidden=10, epochs=40,
                 lr=0.02, seed=11, n_posterior=400)
    exact_mean = X_O[0] / (1.0 + SIG ** 2)
    exact_sd = math.sqrt(SIG ** 2 / (1.0 + SIG ** 2))
    assert abs(res["posterior_mean"][0] - exact_mean) < 0.35
    assert 0.4 * exact_sd < res["posterior_sd"][0] < 2.5 * exact_sd
    assert res["posterior_sd"][0] < 1.0
    assert [h["n_total"] for h in res["history"]] == [40, 80, 120]
    assert res["n_simulations"] == 120
    assert 0.05 < res["acceptance"] < 0.95


def test_a_tight_prior_dominates():
    res = abcnnt(_sim, X_O, lambda t: -0.5 * ((t[0] - 3.0) / 0.05) ** 2,
                 [3.0], n_rounds=2, n_per_round=30, n_layers=2,
                 hidden=10, epochs=30, lr=0.02, seed=4, n_posterior=200,
                 mcmc_step=0.1)
    assert abs(res["posterior_mean"][0] - 3.0) < 0.4


def test_metropolis_recovers_a_known_target():
    draws, rate = mcmc_sample(lambda t: -0.5 * (t[0] - 2.0) ** 2, [0.0],
                              4000, burn=500, step=1.5, seed=2)
    m = sum(p[0] for p in draws) / len(draws)
    sd = math.sqrt(sum((p[0] - m) ** 2 for p in draws) / (len(draws) - 1))
    assert abs(m - 2.0) < 0.15 and abs(sd - 1.0) < 0.15


def test_validation():
    f = MAF(1, 1, n_layers=1, hidden=6, seed=1)
    D = [([0.0], [0.1])]
    for call in (lambda: MAF(0, 1),
                 lambda: MAF(1, 1, n_layers=0),
                 lambda: MAF(1, 1, hidden=0),
                 lambda: train_flow(f, []),
                 lambda: train_flow(f, D, epochs=0),
                 lambda: train_flow(f, D, lr=0.0),
                 lambda: mcmc_sample(_log_prior, [0.0], 0),
                 lambda: mcmc_sample(_log_prior, [0.0], 10, step=0.0),
                 lambda: abcnnt(_sim, [], _log_prior, [0.0]),
                 lambda: abcnnt(_sim, X_O, _log_prior, []),
                 lambda: abcnnt(_sim, X_O, _log_prior, [0.0],
                                n_rounds=0),
                 lambda: abcnnt(_sim, X_O, _log_prior, [0.0],
                                n_per_round=0)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_alias():
    assert sequential_neural_likelihood is abcnnt
