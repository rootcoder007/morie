"""Tests for tmlcic -- adaptive pre-specification in a cluster trial.

Replaces a generated test that called a stub returning mean(y). The
estimand is the SAMPLE effect, so the truth needs no estimating: the
generator produces Y1 and Y0 for every community and the SATE is their
mean difference. Only the randomization varies across replicates, which
is the distribution the inference claims to be valid under.

Full anchor, including the 200-randomization precision comparison:
ledger/wave3/anchor_tmlcic.py.
"""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn import _s03core as k
from morie.fn.tmlcic import (_cv_folds, adaptive_prespecification,
                             candidate_tmle, default_library,
                             influence_curve, tmle_cluster_ic,
                             variance_estimate)

NPAIR = 40
N = 2 * NPAIR


def expit(z):
    return 1.0 / (1.0 + math.exp(-z))


def logit(p):
    return math.log(p / (1.0 - p))


@pytest.fixture(scope="module")
def trial():
    """Communities paired within region.

    The three covariates play the three roles the chapter distinguishes:
    region predicts the outcome and the DESIGN balances it, prevalence
    predicts the outcome and the design does not balance it, and W3 does
    neither. Those roles are asserted below rather than assumed -- an
    earlier version of this fixture used n = 30, where the "noise"
    covariate correlated with the outcome by chance and the selector
    was right to prefer it.
    """
    rng = np.random.default_rng(4)
    region, W1, W3, pair = [], [], [], []
    for j in range(NPAIR):
        r = float(j % 4)
        for _ in range(2):
            region.append(r)
            W1.append(0.50 * float(rng.uniform()))
            W3.append(rng.standard_normal())
            pair.append("pair%02d" % j)
    e = [0.15 * rng.standard_normal() for _ in range(N)]
    Y0 = [expit(-2.2 + 6.0 * W1[i] + 0.8 * region[i] + e[i])
          for i in range(N)]
    Y1 = [expit(logit(Y0[i]) + 0.6) for i in range(N)]
    return {
        "W": [[W1[i], region[i], W3[i]] for i in range(N)],
        "W1": W1, "region": region, "W3": W3, "pair": pair,
        "Y0": Y0, "Y1": Y1,
        "groups": [[2 * j, 2 * j + 1] for j in range(NPAIR)],
        "sate": sum(Y1[i] - Y0[i] for i in range(N)) / N,
    }


def randomize(seed):
    """One treated community per matched pair, as the design does."""
    rr = np.random.default_rng(seed)
    A = [0.0] * N
    for j in range(NPAIR):
        A[2 * j + (0 if float(rr.uniform()) < 0.5 else 1)] = 1.0
    return A


def observed(t, A):
    return [A[i] * t["Y1"][i] + (1.0 - A[i]) * t["Y0"][i]
            for i in range(N)]


CANDS = {
    "unadjusted": {"name": "unadjusted", "cols": (), "interact": False},
    "prevalence": {"name": "W1", "cols": (0,), "interact": False},
    "region": {"name": "W2", "cols": (1,), "interact": False},
    "noise": {"name": "W3", "cols": (2,), "interact": False},
}


def fixed_estimate(t, A, y, cand, design="matched", target="SATE"):
    q1, q0, qa, info = candidate_tmle(y, A, t["W"], cand,
                                      lambda _i: 0.5)
    rows = list(range(N))
    psi = sum(q1[i] - q0[i] for i in rows) / N
    D = influence_curve(y, A, q1, q0, qa, info["gA"], rows, psi, target)
    var, vinfo = variance_estimate(D, y, qa, t["groups"], N, design,
                                   target)
    return psi, math.sqrt(var), vinfo


def test_the_covariates_play_the_roles_the_test_assumes(trial):
    t = trial
    assert all(t["region"][2 * j] == t["region"][2 * j + 1]
               for j in range(NPAIR))
    assert max(abs(t["W1"][2 * j] - t["W1"][2 * j + 1])
               for j in range(NPAIR)) > 0.05
    assert abs(k.corr(t["W1"], t["Y0"])) > 0.4
    assert abs(k.corr(t["W3"], t["Y0"])) < 0.15


def test_it_estimates_the_sample_effect(trial):
    t = trial
    A = randomize(100)
    r = tmle_cluster_ic(observed(t, A), A, t["W"], cluster=t["pair"],
                        target="SATE")
    assert abs(r["estimate"] - t["sate"]) < 2.5 * r["se"]
    assert r["ci"][0] <= t["sate"] <= r["ci"][1]
    assert r["independent_units"] == NPAIR
    assert r["unit"] == "pair"


def test_the_unadjusted_arm_is_the_unadjusted_tmle(trial):
    t = trial
    A = randomize(100)
    y = observed(t, A)
    r = tmle_cluster_ic(y, A, t["W"], cluster=t["pair"])
    r_off = tmle_cluster_ic(y, A, t["W"], cluster=t["pair"],
                            adapt=False)
    assert r_off["estimate"] == pytest.approx(r["unadjusted"], abs=1e-12)


def test_folds_never_split_a_pair(trial):
    t = trial
    folds = _cv_folds(t["groups"], None, "matched", N)
    assert len(folds) == NPAIR
    for j in range(NPAIR):
        assert any(2 * j in f and 2 * j + 1 in f for f in folds)


def test_the_matched_loss_does_not_credit_a_matched_on_covariate(trial):
    """The trap of Sec. 13.3: region is strongly predictive but the
    design already balanced it, so the unmatched loss would select it
    and buy nothing. Eq. (13.8)/(13.9) subtract exactly that."""
    t = trial
    A = randomize(100)
    y = observed(t, A)
    sel_m = adaptive_prespecification(y, A, t["W"], t["groups"],
                                      "matched", "SATE")
    sel_u = adaptive_prespecification(y, A, t["W"], t["groups"],
                                      "unmatched", "SATE")
    rm = dict(zip(sel_m["q_names"], sel_m["q_risks"]))
    ru = dict(zip(sel_u["q_names"], sel_u["q_risks"]))
    gain_m = rm["W2"] / rm["unadjusted"]
    gain_u = ru["W2"] / ru["unadjusted"]
    assert gain_u < 0.85
    assert abs(gain_m - 1.0) < abs(gain_u - 1.0) / 3.0
    # but the covariate the design did NOT balance is still credited
    assert rm["W1"] < rm["unadjusted"]


def test_adjustment_pays_only_where_the_chapter_says_it_does(trial):
    """Measured over randomizations, which is what "precision" means
    here: gains require a covariate that is predictive AND imbalanced."""
    t = trial
    reps = 60
    est = {nm: [] for nm in CANDS}
    for rep in range(reps):
        A = randomize(2000 + rep)
        y = observed(t, A)
        for nm, cand in CANDS.items():
            est[nm].append(fixed_estimate(t, A, y, cand)[0])
    sd = {nm: k.sd(v) for nm, v in est.items()}
    for nm in CANDS:
        assert abs(k.mean(est[nm]) - t["sate"]) < 0.01
    assert 0.95 < sd["region"] / sd["unadjusted"] < 1.05
    assert 0.90 < sd["noise"] / sd["unadjusted"] < 1.10
    assert sd["prevalence"] / sd["unadjusted"] < 0.60


def test_adaptive_prespecification_finds_the_gain(trial):
    t = trial
    reps = 30
    est, chosen = [], {}
    for rep in range(reps):
        A = randomize(2000 + rep)
        r = tmle_cluster_ic(observed(t, A), A, t["W"],
                            cluster=t["pair"], n_folds=5)
        est.append(r["estimate"])
        chosen[r["q_selected"]] = chosen.get(r["q_selected"], 0) + 1
    assert abs(k.mean(est) - t["sate"]) < 0.01
    assert k.sd(est) < 0.025
    assert chosen.get("W1", 0) + chosen.get("W1 x A", 0) > 0.8 * reps


def test_the_matched_variance_uses_the_within_pair_correlation(trial):
    """Eq. (13.7)."""
    t = trial
    A = randomize(100)
    y = observed(t, A)
    _, se_un, _ = fixed_estimate(t, A, y, CANDS["prevalence"],
                                 design="unmatched", target="PATE")
    _, se_m, vin = fixed_estimate(t, A, y, CANDS["prevalence"],
                                  design="matched", target="PATE")
    assert vin["rho"] > 0.0
    assert se_m < se_un


def test_the_sample_effect_is_no_less_precise(trial):
    """Its influence curve drops the covariate-distribution term."""
    t = trial
    A = randomize(100)
    y = observed(t, A)
    _, se_p, _ = fixed_estimate(t, A, y, CANDS["prevalence"],
                                target="PATE")
    _, se_s, _ = fixed_estimate(t, A, y, CANDS["prevalence"],
                                target="SATE")
    assert se_s <= se_p + 1e-12


def test_the_target_changes_the_variance_not_the_estimate(trial):
    t = trial
    A = randomize(100)
    y = observed(t, A)
    rs = tmle_cluster_ic(y, A, t["W"], cluster=t["pair"],
                         target="SATE")
    rp = tmle_cluster_ic(y, A, t["W"], cluster=t["pair"],
                         target="PATE")
    assert rp["estimate"] == pytest.approx(rs["estimate"], abs=0.02)
    assert rp["se"] != rs["se"]


def test_intervals_cover(trial):
    t = trial
    reps, cov = 40, 0
    for rep in range(reps):
        A = randomize(5000 + rep)
        r = tmle_cluster_ic(observed(t, A), A, t["W"],
                            cluster=t["pair"], n_folds=5)
        if r["ci"][0] <= t["sate"] <= r["ci"][1]:
            cov += 1
    assert cov / reps >= 0.88


def test_the_default_library_always_offers_the_unadjusted_model():
    lib = default_library(3)
    assert lib[0]["name"] == "unadjusted"
    assert lib[0]["cols"] == ()
    assert len(lib) == 7
    assert len(default_library(3, interactions=False)) == 4


def test_argument_checks(trial):
    t = trial
    A = randomize(100)
    y = observed(t, A)
    with pytest.raises(ValueError):
        tmle_cluster_ic(y, A, t["W"], target="nope")
    with pytest.raises(ValueError):
        tmle_cluster_ic(y, A, t["W"], design="nope")
    with pytest.raises(ValueError):
        tmle_cluster_ic(y, A, t["W"], design="matched")
    with pytest.raises(ValueError):
        tmle_cluster_ic(y, [2.0] * N, t["W"])
    with pytest.raises(ValueError):
        tmle_cluster_ic(y, [1.0] * N, t["W"])
    with pytest.raises(ValueError):
        tmle_cluster_ic([0.5] * N, A, t["W"])
    with pytest.raises(ValueError):
        tmle_cluster_ic(y[:-1], A, t["W"])
    with pytest.raises(ValueError):
        tmle_cluster_ic(y, A, t["W"], cluster=t["pair"][:-1])
    with pytest.raises(ValueError):
        tmle_cluster_ic(y, A, t["W"], cluster=["a"] * N,
                        design="matched")


def test_clusters_of_other_sizes_are_allowed(trial):
    t = trial
    A = randomize(100)
    r = tmle_cluster_ic(observed(t, A), A, t["W"],
                        cluster=["c%d" % (i // 4) for i in range(N)],
                        design="clustered", n_folds=5)
    assert r["independent_units"] == N // 4
    assert math.isfinite(r["se"])


# ---------------------------------------------------------------------
# The hierarchical TMLEs -- Balzer, Zheng, van der Laan & Petersen (2019)
# ---------------------------------------------------------------------

from morie.fn.tmlcic import cluster_weights, tmle_hierarchical  # noqa


def _make(J, seed, assign="E-only", interference=0.0):
    """Hierarchical generator. `assign` puts the exposure in Sec. 5's
    two regimes; `interference` breaks the sub-model's no-covariate-
    interference assumption in exactly the way the paper warns about."""
    rng = np.random.default_rng(seed)
    cl = []
    for _ in range(J):
        Nj = 8 + int(float(rng.uniform()) * 17)
        cl.append({"E": float(rng.uniform()), "N": Nj,
                   "W": [rng.standard_normal() for _ in range(Nj)]})
    for c in cl:
        wbar = sum(c["W"]) / c["N"]
        z = -0.3 + 2.0 * (c["E"] - 0.5)
        if assign == "EW":
            z += 1.6 * wbar
        c["A"] = 1.0 if float(rng.uniform()) < expit(z) else 0.0

    def mu(a, c, i):
        others = ((sum(c["W"]) - c["W"][i]) / (c["N"] - 1)
                  if c["N"] > 1 else 0.0)
        return expit(-0.9 + 1.1 * c["W"][i] + 1.4 * (c["E"] - 0.5)
                     - 0.8 * a + interference * others)

    y, A, E, W, lab = [], [], [], [], []
    for j, c in enumerate(cl):
        for i in range(c["N"]):
            y.append(1.0 if float(rng.uniform()) < mu(c["A"], c, i)
                     else 0.0)
            A.append(c["A"])
            E.append([c["E"]])
            W.append([c["W"][i]])
            lab.append("c%03d" % j)

    def cf(a):
        return sum(sum(mu(a, c, i) for i in range(c["N"])) / c["N"]
                   for c in cl) / len(cl)

    return {"y": y, "A": A, "E": E, "W": W, "cluster": lab,
            "truth": cf(1.0) - cf(0.0), "J": len(cl), "n": len(y)}


def test_alpha_weights_default_to_one_over_n_j():
    alpha, groups = cluster_weights(["a", "a", "b", "b", "b"])
    assert alpha == [0.5, 0.5, 1 / 3.0, 1 / 3.0, 1 / 3.0]
    for g in groups:
        assert sum(alpha[i] for i in g) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        cluster_weights(["a", "a"], [0.3, 0.3])
    with pytest.raises(ValueError):
        cluster_weights(["a", "a"], [-0.5, 1.5])


@pytest.fixture(scope="module")
def hier():
    return _make(200, 11, assign="E-only")


def test_both_hierarchical_tmles_recover_the_truth(hier):
    d = hier
    r = tmle_hierarchical(d["y"], d["A"], d["E"], d["W"], d["cluster"])
    for nm in ("cluster", "individual"):
        assert (abs(r["estimate_" + nm] - d["truth"])
                < 3.0 * r["se_" + nm])
        assert abs(r["eic_mean_" + nm]) < 1e-9


def test_the_influence_curves_coincide_when_g_depends_only_on_e(hier):
    """Sec. 5. Compared curve against curve, not just the two point
    estimates -- two estimators can agree on a number by luck."""
    d = hier
    r = tmle_hierarchical(d["y"], d["A"], d["E"], d["W"], d["cluster"])
    assert k.corr(r["influence_curve_cluster"],
                  r["influence_curve_individual"]) > 0.9
    assert (abs(r["estimate_cluster"] - r["estimate_individual"])
            < 1.5 * r["se_cluster"])


def test_covariate_interference_biases_the_sub_model_estimator():
    """The paper's warning: under the restricted model's assumptions the
    individual-level TMLE is more efficient, but where covariate
    interference is actually present it buys that with bias."""
    reps = 25
    bias = {"cluster": [], "individual": []}
    for rep in range(reps):
        d = _make(120, 900 + rep, assign="EW", interference=2.5)
        r = tmle_hierarchical(d["y"], d["A"], d["E"], d["W"],
                              d["cluster"])
        for nm in ("cluster", "individual"):
            bias[nm].append(r["estimate_" + nm] - d["truth"])
    mi = k.mean(bias["individual"])
    se_i = k.sd(bias["individual"]) / math.sqrt(reps)
    assert abs(mi) > 3.0 * se_i
    assert abs(mi) > 1.5 * abs(k.mean(bias["cluster"]))


def test_unequal_cluster_sizes_each_count_once(hier):
    d = hier
    r = tmle_hierarchical(d["y"], d["A"], d["E"], d["W"], d["cluster"])
    assert min(r["cluster_sizes"]) < max(r["cluster_sizes"])
    assert len(r["cluster_sizes"]) == d["J"]
    n0 = r["cluster_sizes"][0]
    assert r["cluster_outcome"][0] == pytest.approx(
        sum(d["y"][:n0]) / n0)


def test_a_single_arm_can_be_run(hier):
    d = hier
    both = tmle_hierarchical(d["y"], d["A"], d["E"], d["W"],
                             d["cluster"])
    only = tmle_hierarchical(d["y"], d["A"], d["E"], d["W"],
                             d["cluster"], arm="cluster")
    assert only["arm_reported"] == "cluster"
    assert only["estimate"] == pytest.approx(both["estimate_cluster"],
                                             abs=1e-12)
    assert "estimate_individual" not in only


def test_hierarchical_argument_checks(hier):
    d = hier
    with pytest.raises(ValueError):
        tmle_hierarchical(d["y"], d["A"], d["E"], d["W"], d["cluster"],
                          arm="nope")
    with pytest.raises(ValueError):        # exposure varies in a cluster
        tmle_hierarchical(d["y"], [1.0 - d["A"][0]] + d["A"][1:],
                          d["E"], d["W"], d["cluster"])
    with pytest.raises(ValueError):        # E varies in a cluster
        tmle_hierarchical(d["y"], d["A"], [[9.0]] + d["E"][1:], d["W"],
                          d["cluster"])
    with pytest.raises(ValueError):        # outcome outside [0,1]
        tmle_hierarchical([2.0] + d["y"][1:], d["A"], d["E"], d["W"],
                          d["cluster"])
    with pytest.raises(ValueError):
        tmle_hierarchical(d["y"], [2.0] * d["n"], d["E"], d["W"],
                          d["cluster"])
    with pytest.raises(ValueError):
        tmle_hierarchical(d["y"], [1.0] * d["n"], d["E"], d["W"],
                          d["cluster"])
    with pytest.raises(ValueError):
        tmle_hierarchical(d["y"][:-1], d["A"], d["E"], d["W"],
                          d["cluster"])
    with pytest.raises(ValueError):
        tmle_hierarchical(d["y"], d["A"], d["E"], d["W"], d["cluster"],
                          trim=0.9)
