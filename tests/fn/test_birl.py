"""Tests for birl (Ramachandran & Amir 2007, Bayesian IRL)."""

import math

from morie.fn.birl import (PRIORS, bayesian_irl, birl, log_likelihood,
                           log_prior, policy_iteration, policy_values,
                           policy_walk, q_values)

N, GAMMA = 5, 0.9


def _chain():
    T = []
    for s in range(N):
        left, right = [0.0] * N, [0.0] * N
        left[max(s - 1, 0)] = 1.0
        right[min(s + 1, N - 1)] = 1.0
        T.append([left, right])
    return T


T = _chain()
R_TRUE = [0.0, 0.0, 0.0, 0.0, 1.0]
OBS_RIGHT = [(s, 1) for s in range(N - 1)]


def test_equation_4_and_its_linearity():
    pi = [1] * N
    V = policy_values(T, R_TRUE, GAMMA, pi)
    for s in range(N):
        rhs = R_TRUE[s] + GAMMA * sum(T[s][pi[s]][j] * V[j]
                                      for j in range(N))
        assert abs(V[s] - rhs) < 1e-10
    Ra = [0.3, -0.2, 0.5, 0.1, 1.0]
    Rb = [1.0, 0.4, -0.7, 0.2, 0.0]
    Va = policy_values(T, Ra, GAMMA, pi)
    Vb = policy_values(T, Rb, GAMMA, pi)
    Vc = policy_values(T, [2 * Ra[i] - 3 * Rb[i] for i in range(N)],
                       GAMMA, pi)
    assert max(abs(Vc[i] - (2 * Va[i] - 3 * Vb[i]))
               for i in range(N)) < 1e-10
    assert abs(V[N - 1] - 1.0 / (1.0 - GAMMA)) < 1e-9


def test_policy_iteration_is_optimal():
    got = policy_iteration(T, R_TRUE, GAMMA)
    assert got["policy"] == [1] * N
    Q = got["Q"]
    assert all(Q[s][got["policy"][s]] >= max(Q[s]) - 1e-10
               for s in range(N))
    assert policy_iteration(T, [1.0, 0, 0, 0, 0],
                            GAMMA)["policy"] == [0] * N
    warm = policy_iteration(T, R_TRUE, GAMMA, policy=[1] * N)
    assert warm["sweeps"] == 1


def test_the_boltzmann_likelihood():
    Q = policy_iteration(T, R_TRUE, GAMMA)["Q"]
    ll = log_likelihood(Q, OBS_RIGHT, alpha=1.0)
    by_hand = 0.0
    for s, a in OBS_RIGHT:
        m = max(Q[s])
        by_hand += Q[s][a] - (m + math.log(sum(math.exp(v - m)
                                               for v in Q[s])))
    assert abs(ll - by_hand) < 1e-12
    tot = sum(math.exp(log_likelihood(Q, [(2, a)], alpha=1.0))
              for a in range(2))
    assert abs(tot - 1.0) < 1e-12
    assert abs(math.exp(log_likelihood(Q, [(2, 1)], alpha=1e-6)) -
               0.5) < 1e-4
    assert math.exp(log_likelihood(Q, [(2, 1)], alpha=10.0)) > \
        math.exp(log_likelihood(Q, [(2, 1)], alpha=0.05))


def test_the_priors():
    assert log_prior([0.5, -0.5], "uniform", r_max=1.0) == 0.0
    assert log_prior([2.0, 0.0], "uniform", r_max=1.0) == float("-inf")
    assert abs(log_prior([1.0, -2.0], "gaussian", scale=2.0) +
               5.0 / 8.0) < 1e-15
    assert abs(log_prior([1.0, -2.0], "laplacian", scale=2.0) +
               1.5) < 1e-15
    assert abs(log_prior([1.0, 2.0, 3.0], "ising", J=0.5, H=0.25) +
               (0.5 * 8.0 + 0.25 * 6.0)) < 1e-15


def test_recovery_and_theorem_3():
    res = birl(T, OBS_RIGHT, gamma=GAMMA, n_iter=1500, delta=0.25,
               alpha=5.0, r_max=1.0, seed=7)
    assert max(range(N), key=lambda s: res["reward_mean"][s]) == N - 1
    assert res["policy"] == [1] * N
    assert len(res["reward_sd"]) == N
    assert 0.01 < res["acceptance"] < 0.99


def test_the_mirrored_expert():
    obs_left = [(s, 0) for s in range(1, N)]
    res = birl(T, obs_left, gamma=GAMMA, n_iter=1500, delta=0.25,
               alpha=5.0, r_max=1.0, seed=7)
    assert max(range(N), key=lambda s: res["reward_mean"][s]) == 0
    assert res["policy"] == [0] * N


def test_policy_walk_skips_the_expensive_step():
    res = birl(T, OBS_RIGHT, gamma=GAMMA, n_iter=1500, alpha=5.0,
               r_max=1.0, seed=7)
    assert res["policy_iterations"] / float(res["n_proposals"]) < 0.5


def test_every_prior_runs():
    for p in PRIORS:
        r = birl(T, OBS_RIGHT, gamma=GAMMA, n_iter=400, alpha=5.0,
                 prior=p, r_max=1.0, seed=3)
        assert r["prior"] == p and len(r["reward_mean"]) == N


def test_validation():
    Q = policy_iteration(T, R_TRUE, GAMMA)["Q"]
    bad = [[[0.5, 0.2], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]]
    for call in (lambda: policy_values([], R_TRUE, GAMMA, [1] * N),
                 lambda: policy_values(bad, [0.0, 1.0], GAMMA, [0, 0]),
                 lambda: policy_values(T, R_TRUE, 1.0, [1] * N),
                 lambda: policy_values(T, R_TRUE, GAMMA, [1, 1]),
                 lambda: policy_iteration(T, [0.0], GAMMA),
                 lambda: policy_iteration(T, R_TRUE, GAMMA, policy=[1]),
                 lambda: log_likelihood(Q, []),
                 lambda: log_likelihood(Q, OBS_RIGHT, alpha=0.0),
                 lambda: log_likelihood(Q, [(99, 0)]),
                 lambda: log_likelihood(Q, [(0, 9)]),
                 lambda: log_prior([0.0], "cauchy"),
                 lambda: log_prior([0.0], "gaussian", scale=0.0),
                 lambda: policy_walk(T, OBS_RIGHT, GAMMA, delta=0.0),
                 lambda: policy_walk(T, OBS_RIGHT, GAMMA, n_iter=0),
                 lambda: policy_walk(T, OBS_RIGHT, GAMMA, n_iter=10,
                                     burn=10),
                 lambda: birl(T, OBS_RIGHT, gamma=1.5)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_alias():
    assert bayesian_irl is birl
