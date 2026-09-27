"""Spatial voting utilities and vote probabilities: closed forms and a Monte Carlo check."""

import math

from morie.fn._rng import random_normal
from morie.fn._rrng_core import pt
from morie.fn.svprob import link_cdf, vote_probability
from morie.fn.svutil import voter_utility

X = [[0.2, -0.4], [1.0, 0.5]]
Z = [[0.5, 0.5], [-0.3, 0.1], [1.2, -1.0]]


def test_utility_models_closed_forms():
    x, z = X[0], Z[2]
    d2 = (x[0] - z[0]) ** 2 + (x[1] - z[1]) ** 2
    u = lambda m, **k: voter_utility([x], [z], model=m, **k)["utility"][0][0]  # noqa: E731
    assert u("quadratic") == -d2 and abs(u("linear") + math.sqrt(d2)) < 1e-15
    assert abs(u("cityblock") + abs(x[0] - z[0]) + abs(x[1] - z[1])) < 1e-15
    assert (
        abs(
            u("gaussian", beta=15.0, weights=[0.5, 0.8])
            - 15 * math.exp(-0.5 * (0.25 * (x[0] - z[0]) ** 2 + 0.64 * (x[1] - z[1]) ** 2))
        )
        < 1e-15
    )
    assert abs(u("dot", neutral=[0.1, 0.1]) - ((x[0] - 0.1) * (z[0] - 0.1) + (x[1] - 0.1) * (z[1] - 0.1))) < 1e-15
    cos = (x[0] * z[0] + x[1] * z[1]) / (math.hypot(*x) * math.hypot(*z))
    assert abs(u("angular") - cos) < 1e-15
    assert abs(u("rm", beta=2.0, region=1.0) - (x[0] * z[0] + x[1] * z[1] - 2 * (math.hypot(*z) - 1))) < 1e-15
    assert abs(u("mixed", mix=0.3) - (0.7 * -d2 + 0.3 * (x[0] * z[0] + x[1] * z[1]))) < 1e-15
    eff = [0.1 + 0.4 * (z[0] - 0.1), 0.0 + 0.4 * (z[1] - 0.0)]
    assert abs(u("discount", status_quo=[0.1, 0.0], discount=0.4) + (x[0] - eff[0]) ** 2 + (x[1] - eff[1]) ** 2) < 1e-15
    assert voter_utility([[1, -1, 0]], [[1, 1, 0]], model="categorical_proximity")["utility"] == [[-1.0]]
    assert voter_utility([[1, -1, 0]], [[1, 1, 0]], model="categorical_directional")["utility"] == [[0.0]]


def test_choices_salience_valence_and_totals():
    r = voter_utility(X, Z)
    assert r["choice"] == [1, 0]  # nearest alternative under quadratic loss
    assert r["total"] == [sum(row[j] for row in r["utility"]) for j in range(3)]
    # heavy salience on the first issue changes the second voter's choice
    assert voter_utility(X, Z, weights=[10.0, 0.01])["choice"][1] == 2
    v = voter_utility([[0.0]], [[1.0], [-1.1]], valence=[0.0, 0.5])
    assert v["choice"] == [1] and abs(v["utility"][0][1] - (-1.21 + 0.5)) < 1e-15


def test_binary_links():
    for lk, f in (
        ("normal", lambda u: 0.5 * math.erfc(-u / math.sqrt(2))),
        ("logistic", lambda u: 1 / (1 + math.exp(-u))),
        ("laplace", lambda u: 1 - 0.5 * math.exp(-u) if u >= 0 else 0.5 * math.exp(u)),
        ("gompertz", lambda u: math.exp(-math.exp(-u))),
        ("cloglog", lambda u: 1 - math.exp(-math.exp(u))),
    ):
        for u in (-1.7, 0.0, 0.4):
            assert abs(link_cdf(u, lk) - f(u)) < 1e-15
    assert abs(link_cdf(1.3, "student_t", 4) - float(pt(1.3, 4))) < 1e-15
    p = vote_probability(X, Z[:2], link="logistic", scale=0.5)["probability"]
    U = voter_utility(X, Z[:2])["utility"]
    assert all(
        abs(p[i][0] - 1 / (1 + math.exp(-(U[i][0] - U[i][1]) / 0.5))) < 1e-15 and abs(sum(p[i]) - 1) < 1e-15
        for i in range(2)
    )
    # NOMINATE: Gaussian utility with beta and a normal link
    g = vote_probability(X, Z[:2], model="gaussian", beta=15.0)["probability"][0][0]
    e = [math.exp(-0.5 * ((X[0][0] - z[0]) ** 2 + (X[0][1] - z[1]) ** 2)) for z in Z[:2]]
    assert abs(g - 0.5 * math.erfc(-15 * (e[0] - e[1]) / math.sqrt(2))) < 1e-15


def test_multinomial_and_exponential_rules():
    p = vote_probability(X, Z, rule="multinomial", scale=0.8)["probability"]
    U = voter_utility(X, Z)["utility"]
    for i in range(2):
        den = sum(math.exp(u / 0.8) for u in U[i])
        assert all(abs(p[i][j] - math.exp(U[i][j] / 0.8) / den) < 1e-15 for j in range(3))
    q = vote_probability(X, Z, rule="exponential", scale=2.0)["probability"]
    for i in range(2):
        e = [math.exp(-math.dist(X[i], z) / 2.0) for z in Z]
        assert all(abs(q[i][j] - e[j] / sum(e)) < 1e-15 for j in range(3))


def test_uncertain_ideal_point_integral():
    x, z1, z2 = [0.3, -0.2], [0.5, 0.5], [-0.3, 0.1]
    S = [[0.2, 0.05], [0.05, 0.1]]
    p = vote_probability([x], [z1, z2], ideal_cov=S, scale=0.7)["probability"][0][0]
    dz = [z1[0] - z2[0], z1[1] - z2[1]]
    var = sum(dz[a] * S[a][b] * dz[b] for a in range(2) for b in range(2))
    mean = -((x[0] - z1[0]) ** 2 + (x[1] - z1[1]) ** 2) + (x[0] - z2[0]) ** 2 + (x[1] - z2[1]) ** 2
    assert abs(p - 0.5 * math.erfc(-mean / math.sqrt(0.49 + 4 * var) / math.sqrt(2))) < 1e-15
    # Monte Carlo over the ideal point agrees to its standard error
    e = [float(v) for v in random_normal(40000, seed=3, stream=0)]
    a = math.sqrt(0.2)
    b = 0.05 / a
    c = math.sqrt(0.1 - b * b)
    acc = 0.0
    for k in range(20000):
        xx = [x[0] + a * e[2 * k], x[1] + b * e[2 * k] + c * e[2 * k + 1]]
        du = -((xx[0] - z1[0]) ** 2 + (xx[1] - z1[1]) ** 2) + (xx[0] - z2[0]) ** 2 + (xx[1] - z2[1]) ** 2
        acc += 0.5 * math.erfc(-du / 0.7 / math.sqrt(2))
    assert abs(acc / 20000 - p) < 0.01
