"""Tests for bayhmc (Hoffman & Gelman 2014, NUTS)."""

import math

from morie.fn.bayhmc import (bayhmc, dual_averaging_update,
                             find_reasonable_epsilon, hmc_nuts, leapfrog,
                             no_u_turn)


def logp_std(t):
    return -0.5 * sum(v * v for v in t)


def grad_std(t):
    return [-v for v in t]


RHO = 0.8
INV = [[1.0 / (1 - RHO ** 2), -RHO / (1 - RHO ** 2)],
       [-RHO / (1 - RHO ** 2), 1.0 / (1 - RHO ** 2)]]


def logp_corr(t):
    return -0.5 * sum(t[i] * INV[i][j] * t[j]
                      for i in range(2) for j in range(2))


def grad_corr(t):
    return [-sum(INV[i][j] * t[j] for j in range(2)) for i in range(2)]


def test_leapfrog_is_reversible():
    theta0, r0 = [0.7, -0.3], [0.2, 1.1]
    t, r = theta0, r0
    for _ in range(25):
        t, r = leapfrog(t, r, 0.1, grad_std)
    t, r = list(t), [-x for x in r]
    for _ in range(25):
        t, r = leapfrog(t, r, 0.1, grad_std)
    assert max(abs(t[i] - theta0[i]) for i in range(2)) < 1e-10
    assert max(abs(-r[i] - r0[i]) for i in range(2)) < 1e-10


def test_the_energy_error_is_second_order():
    def err(eps):
        t, r = [1.0, 0.5], [0.3, -0.7]
        h0 = logp_std(t) - 0.5 * sum(v * v for v in r)
        for _ in range(int(4.0 / eps)):
            t, r = leapfrog(t, r, eps, grad_std)
        return abs((logp_std(t) - 0.5 * sum(v * v for v in r)) - h0)

    ratio = err(0.2) / err(0.1)
    assert 3.0 < ratio < 5.0


def test_the_u_turn_rule():
    assert no_u_turn([0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0])
    assert not no_u_turn([0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0])
    assert not no_u_turn([0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [-1.0, 0.0])
    assert no_u_turn([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0])


def test_find_reasonable_epsilon_brackets_a_half():
    def fixed():
        return 0.5

    def accept_at(eps):
        theta = [0.0, 0.0]
        r = [0.5, 0.5]
        t2, r2 = leapfrog(theta, r, eps, grad_std)
        lr = ((logp_std(t2) - 0.5 * sum(v * v for v in r2)) -
              (logp_std(theta) - 0.5 * sum(v * v for v in r)))
        return math.exp(min(lr, 700.0))

    eps = find_reasonable_epsilon([0.0, 0.0], logp_std, grad_std, fixed)
    assert 0.0 < eps < 10.0
    here = accept_at(eps)
    assert ((accept_at(eps * 2.0) - 0.5) * (here - 0.5) <= 0 or
            (accept_at(eps / 2.0) - 0.5) * (here - 0.5) <= 0)
    assert find_reasonable_epsilon([0.0, 0.0], logp_std, grad_std, fixed,
                                   eps=1e-6) > 1e-4
    assert find_reasonable_epsilon([0.0, 0.0], logp_std, grad_std, fixed,
                                   eps=100.0) < 10.0


def test_equation_6():
    mu = math.log(10 * 0.5)
    eps, h, le = dual_averaging_update(1, 0.0, 0.0, 0.65 - 0.4, mu)
    eta = 1.0 / 11.0
    want_h = eta * (0.65 - 0.4)
    assert abs(h - want_h) < 1e-15
    assert abs(eps - math.exp(mu - math.sqrt(1) / 0.05 * want_h)) < 1e-12
    assert abs(le - math.log(eps)) < 1e-12
    up = dual_averaging_update(5, 0.0, 0.0, 0.65 - 0.9, mu)[0]
    down = dual_averaging_update(5, 0.0, 0.0, 0.65 - 0.4, mu)[0]
    assert up > down


def test_the_standard_normal_is_recovered():
    res = bayhmc(logp_std, [0.0, 0.0], n_iter=2000, grad=grad_std, seed=7)
    assert all(abs(v) < 0.12 for v in res["mean"])
    assert all(abs(s - 1.0) < 0.12 for s in res["sd"])
    tail = sum(1 for x in res["samples"] if abs(x[0]) > 1.96) / \
        float(res["n_samples"])
    assert 0.02 < tail < 0.09


def test_adaptation_hits_the_target_acceptance():
    a = bayhmc(logp_std, [0.0, 0.0], n_iter=2000, grad=grad_std, seed=7)
    b = bayhmc(logp_std, [0.0, 0.0], n_iter=1500, grad=grad_std,
               delta=0.9, seed=7)
    assert abs(a["acceptance"] - 0.65) < 0.12
    assert abs(b["acceptance"] - 0.9) < 0.12
    assert b["eps"] < a["eps"]
    assert len(set(a["eps_trace"][a["warmup"]:])) == 1


def test_a_correlated_gaussian():
    res = bayhmc(logp_corr, [0.0, 0.0], n_iter=3000, grad=grad_corr,
                 seed=11)
    xs = res["samples"]
    n = float(len(xs))
    m = [sum(p[i] for p in xs) / n for i in range(2)]
    cov = sum((p[0] - m[0]) * (p[1] - m[1]) for p in xs) / (n - 1)
    assert abs(cov - RHO) < 0.12


def test_nuts_adapts_its_depth_and_hmc_does_not():
    res = bayhmc(logp_std, [0.0, 0.0], n_iter=1500, grad=grad_std, seed=7)
    assert len(set(res["depths"])) > 1
    hmc = bayhmc(logp_std, [0.0, 0.0], n_iter=1500, grad=grad_std,
                 sampler="hmc", n_steps=12, seed=7)
    assert set(hmc["depths"]) == {12}
    assert all(abs(v) < 0.15 for v in hmc["mean"])
    assert all(abs(s - 1.0) < 0.15 for s in hmc["sd"])


def test_a_numerical_gradient_works():
    res = bayhmc(logp_std, [0.0, 0.0], n_iter=400, seed=7)
    assert all(abs(v) < 0.3 for v in res["mean"])


def test_validation():
    for call in (lambda: bayhmc(logp_std, []),
                 lambda: bayhmc(logp_std, [0.0], n_iter=0),
                 lambda: bayhmc(logp_std, [0.0], sampler="gibbs"),
                 lambda: bayhmc(logp_std, [0.0], delta=0.0),
                 lambda: bayhmc(logp_std, [0.0], delta=1.0),
                 lambda: bayhmc(logp_std, [0.0], eps=0.0),
                 lambda: bayhmc(logp_std, [0.0], max_depth=0),
                 lambda: bayhmc(logp_std, [0.0], n_steps=0),
                 lambda: bayhmc(logp_std, [0.0], n_iter=10, warmup=11),
                 lambda: dual_averaging_update(0, 0.0, 0.0, 0.1, 1.0),
                 lambda: dual_averaging_update(1, 0.0, 0.0, 0.1, 1.0,
                                               gamma=0.0),
                 lambda: dual_averaging_update(1, 0.0, 0.0, 0.1, 1.0,
                                               kappa=0.4)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_alias():
    assert hmc_nuts is bayhmc
