"""Tests for tmldta -- data-adaptive target parameters.

Replaces a generated test that called a stub returning mean(y).

The chapter's central claim is one-sided and structural: under a null
where the exposure does nothing, the naive substitution estimator (9.5)
"will always be positively biased (it is always >= 0)". It is tested as
stated -- every replicate, not on average. Full anchor, including the
coverage comparison: ledger/wave3/anchor_tmldta.py.
"""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn import _s03core as k
from morie.fn.tmldta import (discover_levels, split_specific_tmle,
                             tmle_data_adaptive, variable_importance)

LEVELS = [0.0, 1.0, 2.0, 3.0]


def expit(z):
    return 1.0 / (1.0 + math.exp(-z))


def draw(n, seed, effect):
    """At effect = 0 the exposure does nothing at all, so the true
    contrast is exactly zero at EVERY pair of levels -- which is what
    makes any positive number the naive arm reports pure bias."""
    rng = np.random.default_rng(seed)
    W1 = [rng.standard_normal() for _ in range(n)]
    W2 = [rng.standard_normal() for _ in range(n)]
    A = [float(LEVELS[int(float(rng.uniform()) * len(LEVELS))])
         for _ in range(n)]

    def mu(a, i):
        # non-monotone in a, so the argmax is not the largest level
        shape = (a - 1.0) * (3.0 - a) / 2.0
        return expit(-0.4 + 0.9 * W1[i] + 0.5 * W2[i] + effect * shape)

    y = [1.0 if float(rng.uniform()) < mu(A[i], i) else 0.0
         for i in range(n)]
    means = {a: sum(mu(a, i) for i in range(n)) / n for a in LEVELS}
    aL = min(LEVELS, key=lambda a: means[a])
    aH = max(LEVELS, key=lambda a: means[a])
    return {"y": y, "A": A, "W": [[W1[i], W2[i]] for i in range(n)],
            "n": n, "means": means, "aL": aL, "aH": aH,
            "truth": means[aH] - means[aL]}


@pytest.fixture(scope="module")
def null():
    return [draw(400, 1000 + rep, effect=0.0) for rep in range(20)]


@pytest.fixture(scope="module")
def real():
    return draw(1500, 7, effect=2.0)


def test_the_null_design_really_is_null(null):
    d = null[0]
    assert (max(d["means"].values())
            - min(d["means"].values())) < 1e-12


def test_the_naive_estimator_is_never_negative_under_the_null(null):
    """Eq. (9.5) is the max minus the min of one noisy surface, so it is
    structurally one-sided -- a winner's curse with an interval on it."""
    est = [tmle_data_adaptive(d["y"], d["A"], d["W"], method="naive",
                              n_folds=3)["estimate"] for d in null]
    assert all(v >= 0.0 for v in est)
    assert k.mean(est) > 3.0 * k.sd(est) / math.sqrt(len(est))


@pytest.mark.parametrize("method", ["cv-tmle", "sample-split"])
def test_the_split_arms_are_unbiased_under_the_null(null, method):
    est = [tmle_data_adaptive(d["y"], d["A"], d["W"], method=method,
                              n_folds=3)["estimate"] for d in null]
    se = k.sd(est) / math.sqrt(len(est))
    assert abs(k.mean(est)) < 3.0 * se
    # and unlike the naive arm it takes both signs
    assert 0 < sum(1 for v in est if v < 0.0) < len(est)


def test_the_search_is_not_trivial(real):
    """If the true argmax were just the largest level, recovering it
    would prove nothing about the search."""
    assert real["aH"] != max(LEVELS)
    assert real["aL"] != real["aH"]


def test_discover_levels_finds_the_true_argmin_and_argmax(real):
    """Eq. (9.2)-(9.3)."""
    aL, aH, info = discover_levels(real["y"], real["A"], real["W"],
                                   LEVELS)
    assert (aL, aH) == (real["aL"], real["aH"])
    assert info["spread"] > 0.0


def test_it_recovers_the_contrast_at_the_discovered_levels(real):
    r = tmle_data_adaptive(real["y"], real["A"], real["W"], n_folds=3)
    assert abs(r["estimate"] - real["truth"]) < 3.0 * r["se"]
    assert r["ci"][0] <= real["truth"] <= r["ci"][1]
    assert r["modal_levels"] == (real["aL"], real["aH"])


def test_the_split_tmle_solves_its_score_equation(real):
    """Eq. (9.16)."""
    rows = list(range(real["n"]))
    psi, D, info = split_specific_tmle(real["y"], real["A"], real["W"],
                                       LEVELS, real["aL"], real["aH"],
                                       rows, rows)
    assert abs(sum(D.values()) / len(D)) < 1e-8
    assert abs(info["eps"]) > 1e-6


def test_variable_importance_ranks_by_absolute_effect(real):
    """Anchored on the one contrast known exactly -- the exposure's --
    and on the ordering of the two covariates' generating coefficients
    (W1 at 0.9 against W2 at 0.5)."""
    def bin3(v):
        return -1.0 if v < -0.5 else (1.0 if v > 0.5 else 0.0)

    vi = variable_importance(
        real["y"],
        [[real["A"][i], bin3(real["W"][i][0]), bin3(real["W"][i][1])]
         for i in range(real["n"])],
        method="cv-tmle", n_folds=3,
        names=["exposure", "W1", "W2"])
    assert [v["rank"] for v in vi] == [1, 2, 3]
    assert vi[0]["variable"] == "exposure"
    assert abs(vi[0]["estimate"] - real["truth"]) < 0.15
    names = [v["variable"] for v in vi]
    assert names.index("W1") < names.index("W2")
    for v in vi:
        assert v["ci"][0] < v["estimate"] < v["ci"][1]


def test_a_tie_is_flagged_and_separation_is_not(null, real):
    """The levels are an argmin and an argmax, so at a tie the index is
    not unique and the parameter is non-regular. Measured across n the
    interval under-covers there and worsens as n grows, so the module
    reports the regime instead of leaving it to be discovered."""
    d = null[0]
    r_tie = tmle_data_adaptive(d["y"], d["A"], d["W"], n_folds=3)
    assert r_tie["near_tie"]
    r_sep = tmle_data_adaptive(real["y"], real["A"], real["W"],
                               n_folds=3)
    assert not r_sep["near_tie"]
    assert r_sep["separation"] > r_tie["separation"]
    assert r_sep["level_agreement"] == 1.0


def test_a_separated_contrast_is_covered_at_the_nominal_rate(real):
    """Coverage belongs where the chapter's conditions hold."""
    reps, cov = 20, 0
    bias = []
    for rep in range(reps):
        d = draw(600, 7000 + rep, effect=2.0)
        r = tmle_data_adaptive(d["y"], d["A"], d["W"], n_folds=3)
        bias.append(r["estimate"] - d["truth"])
        if r["ci"][0] <= d["truth"] <= r["ci"][1]:
            cov += 1
    assert cov >= 0.80 * reps
    assert abs(k.mean(bias)) < 0.05


def test_argument_checks(real):
    d = real
    with pytest.raises(ValueError):
        tmle_data_adaptive(d["y"], d["A"], d["W"], method="nope")
    with pytest.raises(ValueError):
        tmle_data_adaptive(d["y"], [1.0] * d["n"], d["W"])
    with pytest.raises(ValueError):
        tmle_data_adaptive(d["y"], d["A"], d["W"],
                           candidate_strata=[0.0, 1.0, 99.0])
    with pytest.raises(ValueError):
        tmle_data_adaptive(d["y"][:-1], d["A"], d["W"])
    with pytest.raises(ValueError):
        tmle_data_adaptive(d["y"], d["A"], d["W"][:-1])
    with pytest.raises(ValueError):
        tmle_data_adaptive([1.0] * d["n"], d["A"], d["W"])
    with pytest.raises(ValueError):
        tmle_data_adaptive(d["y"], d["A"], d["W"], bounds=(0.0, 0.5))
    with pytest.raises(ValueError):
        tmle_data_adaptive(d["y"], d["A"], d["W"], trim=0.7)
    with pytest.raises(ValueError):
        tmle_data_adaptive(d["y"][:5], d["A"][:5], d["W"][:5])
    with pytest.raises(ValueError):
        variable_importance(d["y"], [[a] for a in d["A"]])
    with pytest.raises(ValueError):
        variable_importance(
            d["y"], [[d["A"][i], round(d["W"][i][0])]
                     for i in range(d["n"])], names=["only-one"])
