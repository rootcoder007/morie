"""Horowitz Sec. 4.3.3-4.4 max-score extensions: choice-based samples,
panel data with fixed effects, ordered response, and the rate."""

from morie.fn import _array_core as np
import pytest

from morie.fn.hrzcbsm import choice_based_optimal_shares, horowitz_choice_based_sms
from morie.fn.hrzormsc import horowitz_ordered_max_score
from morie.fn.hrzpanms import horowitz_panel_max_score
from morie.fn.hrzsmsrc import horowitz_sms_rate


def _choice_based_sample(rng, n1, n0, beta):
    """Draw a sample stratified ON Y, which is what makes it
    choice-based: X is sampled conditional on Y by rejection."""
    keep = {1: [], 0: []}
    while len(keep[1]) < n1 or len(keep[0]) < n0:
        X = rng.standard_normal((5000, len(beta)))
        y = (X @ beta + rng.standard_normal(5000) > 0).astype(int)
        for lab, want in ((1, n1), (0, n0)):
            need = want - len(keep[lab])
            if need > 0:
                keep[lab].extend(X[y == lab][:need])
    X = np.vstack([np.array(keep[1]), np.array(keep[0])])
    y = np.r_[np.ones(n1), np.zeros(n0)]
    return X, y


def test_choice_based_estimator_recovers_beta_on_a_stratified_sample():
    rng = np.random.default_rng(0)
    beta = np.array([1.0, -0.8])
    # 50/50 by design, while the population share is far from half
    X, y = _choice_based_sample(rng, 700, 700, beta)
    pi1 = 0.5 + np.mean(beta) * 0  # placeholder-free: computed below
    # population P(Y=1) under the true model, by simulation
    Xp = rng.standard_normal((200000, 2))
    pi1 = float(np.mean(Xp @ beta + rng.standard_normal(200000) > 0))
    out = horowitz_choice_based_sms(X, y, pi1)
    assert out["n1"] == 700 and out["n0"] == 700
    assert out["beta"][0] == 1.0
    assert abs(out["beta"][1] - beta[1]) < 0.25
    with pytest.raises(ValueError):
        horowitz_choice_based_sms(X, y, 1.5)
    with pytest.raises(ValueError):
        horowitz_choice_based_sms(X[:, :1], y, pi1)


def test_optimal_shares_follow_the_square_root_rule():
    out = choice_based_optimal_shares(0.1)
    assert out["q1"] == pytest.approx(np.sqrt(0.1) / (np.sqrt(0.1) + np.sqrt(0.9)))
    assert out["q0"] + out["q1"] == pytest.approx(1.0)
    # the optimum genuinely beats both a random sample and the
    # 50/50 split, and it is neither of them
    assert out["factor"] < out["factor_at_random_sample"]
    even = 0.1 / 0.5 + 0.9 / 0.5
    assert out["factor"] < even
    assert out["q1"] != pytest.approx(0.1)
    assert out["q1"] != pytest.approx(0.5)
    # at pi1 = 1/2 the two rules coincide
    assert choice_based_optimal_shares(0.5)["q1"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        choice_based_optimal_shares(0.0)


def _panel_fit(n, seed, beta=np.array([1.0, 0.6]), T=2):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, T, 2))
    U = rng.standard_normal(n) * 3.0  # large and unrestricted
    y = ((X @ beta + U[:, None] + rng.standard_normal((n, T))) > 0).astype(float)
    return horowitz_panel_max_score(X, y, T)


def test_panel_max_score_removes_the_fixed_effect():
    out = _panel_fit(4000, 2)
    assert abs(out["beta"][1] - 0.6) < 0.2
    assert out["intercept_identified"] is False
    assert out["n_pairs"] == 4000
    assert 0 < out["n_discordant_pairs"] < 4000


def test_panel_max_score_converges_as_n_grows():
    # The estimator is consistent but SLOW -- only discordant pairs
    # carry information, and a large fixed effect makes most
    # individuals answer the same way in both periods. Measured
    # medians over five seeds: n=600 -> 0.385 (136 informative pairs),
    # n=2000 -> 0.512, n=8000 -> 0.597 against the true 0.6. A single
    # small sample therefore proves nothing about correctness; the
    # trend does.
    errs = [np.median([abs(_panel_fit(n, s)["beta"][1] - 0.6) for s in range(5)])
            for n in (600, 8000)]
    assert errs[1] < errs[0] / 2


def test_panel_max_score_names_the_columns_it_cannot_identify():
    rng = np.random.default_rng(3)
    n, T = 200, 3
    X = rng.standard_normal((n, T, 3))
    X[:, :, 2] = rng.standard_normal((n, 1))  # constant within individual
    y = (rng.standard_normal((n, T)) > 0).astype(float)
    out = horowitz_panel_max_score(X, y, T, n_restarts=2)
    assert out["unidentified_columns"] == [2]
    assert out["n_pairs"] == n * 3  # C(3, 2) pairs per individual
    with pytest.raises(ValueError):
        horowitz_panel_max_score(X, y, 1)


def test_ordered_max_score_minimises_and_maximising_would_fail():
    rng = np.random.default_rng(0)
    n = 4000
    beta = np.array([1.0, -0.7])
    alpha = np.array([-1.0, 0.0, 1.2])
    X = rng.standard_normal((n, 2))
    ystar = X @ beta + rng.standard_normal(n) * 0.8
    y = np.searchsorted(alpha, ystar)
    out = horowitz_ordered_max_score(X, y, thresholds=alpha)
    assert out["sense"] == "minimised"
    assert out["M"] == 4
    assert out["scale_normalisation_required"] is False
    assert abs(out["beta"][1] - beta[1]) < 0.15

    # the book prints "maximize" over (4.43); maximising the same
    # objective does NOT recover beta, it runs to the boundary
    W = 1.0 + y.astype(float)
    def S(b2):
        v = X @ np.array([1.0, b2])
        return float(np.mean(np.abs(W - (1.0 + np.sum(v[:, None] > alpha, axis=1)))))
    grid = np.linspace(-3, 3, 601)
    vals = np.array([S(g) for g in grid])
    assert abs(grid[vals.argmin()] - beta[1]) < 0.1
    assert abs(grid[vals.argmax()]) == pytest.approx(3.0)


def test_ordered_max_score_estimates_unknown_thresholds():
    rng = np.random.default_rng(5)
    n = 1500
    X = rng.standard_normal((n, 2))
    ystar = X @ np.array([1.0, -0.5]) + rng.standard_normal(n) * 0.7
    y = np.searchsorted(np.array([0.0, 1.0]), ystar)
    out = horowitz_ordered_max_score(X, y, n_restarts=4)
    assert out["thresholds_estimated"] is True
    assert out["thresholds"][0] == 0.0            # Lee's normalisation
    assert np.all(np.diff(out["thresholds"]) > 0)  # stays ordered
    with pytest.raises(ValueError):
        horowitz_ordered_max_score(X, (X[:, 0] > 0).astype(int))  # only 2 categories


def test_sms_rate_is_derived_from_the_theorem_normalisation():
    out = horowitz_sms_rate(10_000, 2)
    assert out["exponent"] == pytest.approx(-2 / 5)
    assert out["bandwidth_exponent"] == pytest.approx(-1 / 5)
    assert out["rate"] == pytest.approx(10_000 ** (-2 / 5))
    # smoothing beats the unsmoothed n^{-1/3} ...
    assert out["rate"] < out["unsmoothed_rate"]
    assert out["ratio_to_unsmoothed"] < 1
    # ... but never reaches n^{-1/2}, for any admissible s
    for s in (2, 3, 5, 50):
        assert horowitz_sms_rate(1000, s)["exponent"] > -0.5
    assert horowitz_sms_rate(1000, 1000)["exponent"] == pytest.approx(-0.5, abs=1e-3)
    assert out["attains_root_n"] is False
    with pytest.raises(ValueError):
        horowitz_sms_rate(1000, 1)
