"""Wittman-Calvert equilibrium of policy-motivated candidates."""

import math

from morie.fn.polmot import expected_utilities, policy_motivated_equilibrium


def test_symmetric_closed_form():
    phi0 = 1 / math.sqrt(2 * math.pi)
    for a, s in ((1.0, 0.5), (2.0, 1.0), (0.5, 0.1)):
        r = policy_motivated_equilibrium([-a], [a], [0.0], s)
        x = a / (1 + 2 * a * phi0 / s)
        assert abs(r.value[0][0] + x) < 1e-12 and abs(r.value[1][0] - x) < 1e-12
        assert abs(r.extra["win_prob_2"] - 0.5) < 1e-12
    # Calvert (1985): certainty forces convergence to the centre
    assert abs(policy_motivated_equilibrium([-1.0], [1.0], [0.0], 1e-4).value[1][0]) < 1e-3


def test_two_dimensional_equilibrium():
    r = policy_motivated_equilibrium([-1.0, 0.0], [1.0, 0.0], [0.0, 0.0], 0.5)
    x = 1 / (1 + 2 * 0.3989422804014327 / 0.5)
    assert abs(r.value[1][0] - x) < 1e-12 and abs(r.value[1][1]) < 1e-12
    a1, a2, mu = [-1.0, 0.5], [1.5, 1.0], [0.2, -0.3]
    r = policy_motivated_equilibrium(a1, a2, mu, 0.7)
    x1, x2 = r.value
    # PolicyMotivatedEquilibrium(c(-1, 0.5), c(1.5, 1), c(0.2, -0.3), 0.7) in R
    ref = [-0.30714734126985294, 0.062815566635104833, 0.71908169996883797, 0.18770075462936342]
    assert all(abs(p - q) < 1e-10 for p, q in zip(x1 + x2, ref))
    assert max(abs(g) for G in r.extra["gradients"] for g in G) < 1e-10
    u1, u2, _ = expected_utilities(x1, x2, a1, a2, mu, 0.7)
    for e in ([1e-3, 0], [0, 1e-3], [-1e-3, 0], [0, -1e-3]):
        assert expected_utilities([p + q for p, q in zip(x1, e)], x2, a1, a2, mu, 0.7)[0] < u1
        assert expected_utilities(x1, [p + q for p, q in zip(x2, e)], a1, a2, mu, 0.7)[1] < u2


def test_plurality_downs_and_eaton_lipsey():
    from morie.fn.plucmp import plurality_competition

    r = plurality_competition([0.0, 1.0, 2.0, 7.0, 9.0])
    assert r.extra["positions"] == [2.0, 2.0] and r.value == [0.5, 0.5] and r.extra["is_equilibrium"]
    # off the median the laggard gains by moving to it
    assert plurality_competition([0.0, 1.0, 2.0, 7.0, 9.0], [1.0, 2.0]).extra["gain"][0] > 0.09
    assert plurality_competition([0.0, 1.0, 2.0, 7.0]).extra["median_interval"] == [1.0, 2.0]
    U = [(i + 0.5) / 120 for i in range(120)]
    # Eaton and Lipsey (1975), uniform electorate: paired quartiles for four
    assert plurality_competition(U, [0.25, 0.25, 0.75, 0.75]).extra["is_equilibrium"]
    assert plurality_competition(U, [1 / 6, 1 / 6, 0.5, 5 / 6, 5 / 6]).extra["is_equilibrium"]
    for conf in ([0.25, 0.5, 0.75], [0.5, 0.5, 0.5], [0.3, 0.3, 0.7]):
        assert not plurality_competition(U, conf).extra["is_equilibrium"]


def logit_voters():
    from morie.fn._rng import random_normal

    Z = [float(t) for t in random_normal(80, seed=3, stream=0)]
    return [[Z[2 * i], 0.6 * Z[2 * i + 1]] for i in range(40)]


def test_logit_competition_schofield_hessian_at_the_mean():
    from morie.fn._mlfa import eigh_desc
    from morie.fn.logcmp import logit_competition

    X = logit_voters()
    r = logit_competition(X, [0.0, 0.5, 1.0], 0.2)
    mean = [sum(x[k] for x in X) / 40 for k in range(2)]
    assert all(abs(z[k] - mean[k]) < 1e-10 for z in r.value for k in range(2)) and r.extra["is_local_nash"]
    for j, rho in enumerate(r.extra["mean_shares"]):
        scale = 2 * 0.2 * rho * (1 - rho)
        want = sorted(scale * e for e in eigh_desc(r.extra["characteristic_matrices"][j])[0])
        assert all(abs(a - b) < 1e-6 for a, b in zip(sorted(r.extra["hessian_eigenvalues"][j]), want))


def test_logit_competition_divergent_equilibrium():
    from morie.fn._mlfa import eigh_desc
    from morie.fn.logcmp import logit_competition

    X = logit_voters()
    r = logit_competition(X, [0.0, 0.2, 1.5], 1.5)
    # the mean fails Schofield's condition for the lowest-valence party
    assert max(eigh_desc(r.extra["characteristic_matrices"][0])[0]) > 0
    assert r.extra["is_local_nash"] and max(abs(t) for g in r.extra["gradients"] for t in g) < 1e-10
    # LogitCompetition(X, c(0, 0.2, 1.5), 1.5)$positions in R
    ref = [[-0.593141209637, -0.1798003389928], [0.864816154602, 0.0849780285912], [0.168313932046, -0.0407506927967]]
    assert all(abs(a - b) < 1e-9 for z, w in zip(r.value, ref) for a, b in zip(z, w))
