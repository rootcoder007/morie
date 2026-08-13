"""Tests for flxipt -- Super Learner and IPTW with an SL propensity.

Replaces a generated test that called a stub returning mean(y). Full
anchor, including the replicate-level comparisons:
ledger/wave3/anchor_flxipt.py.
"""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn import _s03core as k
from morie.fn.flxipt import (_nnls_simplex, _project_simplex, cv_risk,
                             default_learners, flexible_iptw, iptw_ate,
                             super_learner)


def expit(z):
    return 1.0 / (1.0 + math.exp(-z))


def draw(n, seed, truth_in_library):
    rng = np.random.default_rng(seed)
    W = [[rng.standard_normal() for _ in range(3)] for _ in range(n)]
    if truth_in_library:
        mu = [0.4 + 0.9 * W[i][0] - 0.7 * W[i][1] + 0.5 * W[i][2]
              for i in range(n)]
    else:
        mu = [1.2 * math.sin(1.7 * W[i][0]) + 0.8 * abs(W[i][1])
              - 0.6 * math.tanh(2.0 * W[i][2]) for i in range(n)]
    y = [mu[i] + 0.5 * rng.standard_normal() for i in range(n)]
    return {"y": y, "W": W, "mu": mu, "n": n}


def draw_ps(n, seed):
    """The treatment mechanism is NOT a main-terms logistic: an
    interaction and a threshold. The outcome is linear, so the ATE is
    known exactly."""
    rng = np.random.default_rng(seed)
    W = [[rng.standard_normal() for _ in range(3)] for _ in range(n)]
    A, ps = [], []
    for i in range(n):
        z = (-0.2 + 1.3 * W[i][0] * W[i][1]
             + 1.1 * (1.0 if W[i][2] > 0.4 else -1.0))
        p = min(max(expit(z), 0.02), 0.98)
        ps.append(p)
        A.append(1.0 if float(rng.uniform()) < p else 0.0)
    tau = 1.5
    y = [0.3 + 1.0 * W[i][0] + 0.8 * W[i][1] - 0.5 * W[i][2]
         + tau * A[i] + 0.4 * rng.standard_normal() for i in range(n)]
    return {"y": y, "A": A, "W": W, "ps": ps, "tau": tau, "n": n}


def test_the_simplex_projection_is_the_closest_point():
    """Not a clip-and-renormalise: the closest point on the simplex to
    (3, -1, 0) is (1, 0, 0)."""
    assert _project_simplex([0.5, 0.4, 0.1]) == pytest.approx(
        [0.5, 0.4, 0.1])
    pr = _project_simplex([3.0, -1.0, 0.0])
    assert pr == pytest.approx([1.0, 0.0, 0.0])
    assert sum(pr) == pytest.approx(1.0)


def test_nnls_returns_a_convex_combination():
    a = _nnls_simplex([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                      [2.0, 0.0, 2.0])
    assert sum(a) == pytest.approx(1.0)
    assert all(v >= -1e-12 for v in a)


@pytest.mark.parametrize("in_lib", [True, False])
def test_the_ensemble_is_at_least_as_good_as_every_candidate(in_lib):
    """The oracle result. Every vertex of the simplex is feasible, so a
    converged solver can never do worse than the best single candidate
    -- which is exactly how an under-converged one was caught."""
    d = draw(900, 21 if in_lib else 22, in_lib)
    r = super_learner(d["y"], d["W"], n_folds=5, binary=False)
    risks = r["cv_risk"]
    assert r["cv_risk_ensemble"] <= min(risks.values()) + 1e-9
    assert r["cv_risk_ensemble"] < 0.9 * max(risks.values())
    assert sum(r["weight_vector"]) == pytest.approx(1.0, abs=1e-6)
    assert all(v >= -1e-12 for v in r["weight_vector"])


def test_a_correctly_specified_candidate_takes_the_weight():
    d = draw(900, 21, True)
    r = super_learner(d["y"], d["W"], n_folds=5, binary=False)
    assert r["discrete_choice"] == "main"
    assert r["weights"]["main"] > 0.8


def test_complementary_candidates_make_the_combination_strictly_win():
    """'At least as well' can be met by collapsing onto one vertex. The
    case stacking exists for is candidates with complementary errors."""
    rng = np.random.default_rng(55)
    n = 900
    W = [[rng.standard_normal() for _ in range(3)] for _ in range(n)]
    y = [W[i][0] + W[i][1] + W[i][2] + 0.6 * rng.standard_normal()
         for i in range(n)]
    lib = [{"name": "cols01", "kind": "subset", "cols": (0, 1),
            "penalty": 0.0},
           {"name": "cols12", "kind": "subset", "cols": (1, 2),
            "penalty": 0.0},
           {"name": "cols02", "kind": "subset", "cols": (0, 2),
            "penalty": 0.0}]
    r = super_learner(y, W, library=lib, n_folds=5, binary=False)
    assert r["cv_risk_ensemble"] < min(r["cv_risk"].values()) - 1e-6
    assert sum(1 for v in r["weight_vector"] if v > 0.01) >= 2


def test_the_discrete_super_learner_is_the_cv_argmin():
    d = draw(900, 22, False)
    r = super_learner(d["y"], d["W"], n_folds=5, binary=False,
                      meta="discrete")
    best = min(r["cv_risk"], key=lambda nm: r["cv_risk"][nm])
    assert r["weights"][best] == 1.0
    assert sum(r["weight_vector"]) == 1.0


def test_in_sample_level_one_data_understates_its_own_risk():
    """What building Z in-sample reliably breaks is the risk estimate --
    which is what the selection depends on."""
    reps, gaps, wgap = 12, [], []
    for rep in range(reps):
        rng = np.random.default_rng(300 + rep)
        n = 140
        W = [[rng.standard_normal() for _ in range(6)]
             for _ in range(n)]
        y = [1.2 * math.sin(1.7 * W[i][0]) + 0.8 * abs(W[i][1])
             - 0.6 * math.tanh(2.0 * W[i][2])
             + 0.5 * rng.standard_normal() for i in range(n)]
        hon = super_learner(y, W, n_folds=5, binary=False)
        chn = super_learner(y, W, n_folds=5, binary=False,
                            honest_level_one=False)
        gaps.append(hon["cv_risk_ensemble"] - chn["cv_risk_ensemble"])
        wgap.append(max(
            abs(hon["weight_vector"][t] - chn["weight_vector"][t])
            for t in range(len(hon["weight_vector"]))))
    assert all(v > 0.0 for v in gaps)
    assert k.mean(wgap) > 0.05


def test_the_sl_propensity_recovers_a_non_parametric_mechanism():
    d = draw_ps(1500, 41)
    r_sl = iptw_ate(d["y"], d["A"], d["W"], n_folds=5)
    r_main = iptw_ate(d["y"], d["A"], d["W"], n_folds=5,
                      library=[{"name": "main", "kind": "main",
                                "penalty": 0.0}])
    inter = [d["W"][i][0] * d["W"][i][1] for i in range(d["n"])]
    # the parametric fit cannot see the interaction at all
    assert abs(k.corr(r_main["propensity"], inter)) < 0.2
    assert (k.corr(r_sl["propensity"], d["ps"])
            > k.corr(r_main["propensity"], d["ps"]) + 0.15)
    assert r_sl["ci"][0] <= d["tau"] <= r_sl["ci"][1]


def test_the_weights_are_equation_three():
    d = draw_ps(600, 41)
    r = flexible_iptw(d["A"], d["W"], n_folds=5)
    g = r["propensity"]
    for i in range(d["n"]):
        want = d["A"][i] / g[i] + (1.0 - d["A"][i]) / (1.0 - g[i])
        assert r["weights"][i] == pytest.approx(want, abs=1e-12)
    rs = flexible_iptw(d["A"], d["W"], n_folds=5, stabilize=True)
    assert abs(k.mean(rs["weights"]) - 1.0) < 0.25
    assert k.mean(r["weights"]) > 1.6


def test_trimming_bounds_the_propensity():
    d = draw_ps(600, 41)
    r = flexible_iptw(d["A"], d["W"], n_folds=5, trim=0.05)
    assert r["min_propensity"] >= 0.05 - 1e-12
    assert r["max_propensity"] <= 0.95 + 1e-12


def test_the_default_library_offers_the_intercept_fallback():
    assert default_learners(3)[0]["name"] == "intercept"
    assert len(default_learners(1)) == 3


def test_argument_checks():
    d = draw(200, 22, False)
    with pytest.raises(ValueError):
        super_learner(d["y"], d["W"], meta="nope")
    with pytest.raises(ValueError):
        super_learner(d["y"], d["W"], library=[])
    with pytest.raises(ValueError):
        super_learner(d["y"], d["W"],
                      library=[{"name": "x", "kind": "nope",
                                "penalty": 0.0}])
    with pytest.raises(ValueError):
        cv_risk([1.0, 0.0], [[0.5], [0.5]], loss="nope")
    with pytest.raises(ValueError):
        super_learner(d["y"], d["W"], loss="nll")
    with pytest.raises(ValueError):
        super_learner(d["y"][:-1], d["W"])
    with pytest.raises(ValueError):
        super_learner(d["y"][:4], d["W"][:4])
    with pytest.raises(ValueError):
        flexible_iptw([2.0] * 50, [[0.0]] * 50)
    with pytest.raises(ValueError):
        flexible_iptw([1.0] * 50, [[0.0]] * 50)
    with pytest.raises(ValueError):
        flexible_iptw([1.0, 0.0] * 25, [[0.0]] * 50, trim=0.8)
    with pytest.raises(ValueError):
        iptw_ate(d["y"][:-1], [1.0, 0.0] * 100, d["W"])
