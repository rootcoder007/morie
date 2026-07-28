# morie.fn -- test file (rootcoder007/morie)
"""Hastie, Tibshirani and Friedman, *The Elements of Statistical
Learning*, 2nd ed.

Section 7.11 is the spine of this shelf and it is an argument, not a
list of formulas: Err_boot (7.54) is biased LOW because the bootstrap
samples overlap the training set; Err^(1) (7.56) removes the overlap
and is then biased HIGH because a bootstrap sample holds only about
0.632N distinct points; .632 (7.57) corrects that; and .632+ (7.61)
corrects .632 when the rule overfits. Each step is tested as the step
it is, on the book's own worked numbers where it gives them.
"""

import numpy as np
import pytest

from morie.fn._esl import (BOOTSTRAP_INCLUSION_LIMIT, default_mtry,
                           gaussian_product_kernel_density,
                           inclusion_probability)
from morie.fn.eslboo import esl_bootstrap_err
from morie.fn.eslkrn import esl_kernel_density
from morie.fn.eslo63 import esl_oob_632
from morie.fn.eslrft import esl_random_forest
from morie.fn.eslsig import esl_residual_variance


# ----------------------------------------------------------- Ch. 3


def test_residual_variance_is_unbiased_and_the_mle_is_not():
    """(3.8). The N - p - 1 denominator is not a convention, it is
    what makes E(sigma^2_hat) = sigma^2. Averaging over many draws
    from a known model is the direct check, and the same average
    shows the N-denominator version sitting low by exactly
    (N-p-1)/N."""
    rng = np.random.default_rng(11)
    n, p, sigma = 25, 4, 1.7
    beta = np.array([2.0, -1.0, 0.5, 3.0])
    unb, mle = [], []
    for _ in range(4000):
        X = rng.normal(size=(n, p))
        y = 1.0 + X @ beta + rng.normal(scale=sigma, size=n)
        o = esl_residual_variance(X, y)
        unb.append(o["value"])
        mle.append(o["mle_variance"])
    assert np.mean(unb) == pytest.approx(sigma ** 2, rel=0.03)
    assert np.mean(mle) < sigma ** 2
    assert np.mean(mle) / np.mean(unb) == pytest.approx((n - p - 1) / n,
                                                        rel=1e-9)


def test_residual_variance_counts_the_intercept_once():
    """A design that already carries a column of ones must not be
    charged a degree of freedom for it twice."""
    rng = np.random.default_rng(13)
    X = rng.normal(size=(40, 3))
    y = X @ [1.0, 2.0, -1.0] + 5.0 + rng.normal(size=40)
    bare = esl_residual_variance(X, y)
    withone = esl_residual_variance(np.column_stack([np.ones(40), X]), y)
    assert bare["intercept_in_X"] is False
    assert withone["intercept_in_X"] is True
    assert bare["df"] == withone["df"] == 36
    assert bare["value"] == pytest.approx(withone["value"], rel=1e-12)


def test_residual_variance_refuses_an_undefined_estimate():
    """With N <= p + 1 the residual degrees of freedom are zero or
    negative and (3.8) divides by it. Returning inf or nan would be
    worse than refusing."""
    rng = np.random.default_rng(17)
    X = rng.normal(size=(5, 4))
    y = rng.normal(size=5)
    with pytest.raises(ValueError, match="N > p \\+ 1"):
        esl_residual_variance(X, y)
    assert esl_residual_variance(rng.normal(size=(6, 4)),
                                 rng.normal(size=6))["df"] == 1


def test_residual_variance_recovers_least_squares_when_beta_is_omitted():
    rng = np.random.default_rng(19)
    X = rng.normal(size=(50, 2))
    y = X @ [3.0, -2.0] + 1.0 + rng.normal(scale=0.4, size=50)
    D = np.column_stack([np.ones(50), X])
    b = np.linalg.lstsq(D, y, rcond=None)[0]
    assert esl_residual_variance(X, y)["value"] == pytest.approx(
        esl_residual_variance(np.column_stack([np.ones(50), X]), y,
                              beta=b)["value"], rel=1e-12)
    # least squares minimises RSS, so any other beta gives a larger one
    assert esl_residual_variance(X, y)["rss"] <= esl_residual_variance(
        np.column_stack([np.ones(50), X]), y, beta=b + 0.1)["rss"]


# ----------------------------------------------------------- Ch. 6


def test_kernel_density_is_a_density_and_matches_the_convolution_form():
    """(6.23) says the estimate IS the empirical df convolved with a
    Gaussian of standard deviation lambda. Computing that convolution
    directly is an independent route to the same numbers."""
    rng = np.random.default_rng(23)
    data = rng.normal(size=400)
    g = np.linspace(-5, 5, 601)
    o = esl_kernel_density(g, data, 0.4)
    # the grid truncates the tails, so the mass is 1 only to the
    # accuracy of that truncation
    assert o["mass"] == pytest.approx(1.0, abs=1e-4)
    assert np.all(o["density"] >= 0)
    direct = np.array([
        np.mean(np.exp(-0.5 * ((t - data) / 0.4) ** 2)
                / (0.4 * np.sqrt(2 * np.pi))) for t in g])
    assert np.allclose(o["density"], direct, rtol=1e-12)
    assert o["is_convolution"] is True


def test_kernel_density_normaliser_carries_p_over_two():
    """(6.24): the constant is N(2 lambda^2 pi)^{p/2}. In two
    dimensions an exponent of p instead of p/2 would leave the
    estimate off by a factor of 2 lambda^2 pi, and the mass check
    is what sees it."""
    rng = np.random.default_rng(29)
    data = rng.normal(size=(600, 2))
    ax = np.linspace(-4, 4, 90)
    G = np.array([[a, b] for a in ax for b in ax])
    lam = 0.6
    d = gaussian_product_kernel_density(G, data, lam).reshape(90, 90)
    step = ax[1] - ax[0]
    assert d.sum() * step ** 2 == pytest.approx(1.0, abs=0.02)
    o = esl_kernel_density(G[:5], data, lam)
    assert o["p"] == 2
    assert o["normaliser"] == pytest.approx(
        600 * (2 * lam ** 2 * np.pi) ** 1.0, rel=1e-12)


def test_kernel_density_tracks_a_known_density():
    rng = np.random.default_rng(31)
    g = np.linspace(-3, 3, 200)
    o = esl_kernel_density(g, rng.normal(size=4000), 0.25)
    truth = np.exp(-0.5 * g ** 2) / np.sqrt(2 * np.pi)
    assert np.max(np.abs(o["density"] - truth)) < 0.03
    with pytest.raises(ValueError, match="positive"):
        esl_kernel_density(g, rng.normal(size=50), -1.0)


# ----------------------------------------------------------- Ch. 7


def test_inclusion_probability_is_the_exact_finite_n_value():
    """(7.55). The .632 in the estimator's name is this number, and
    the exact value is NOT 0.632 at the sample sizes where anyone
    reaches for a bootstrap."""
    assert inclusion_probability(10) == pytest.approx(1 - 0.9 ** 10, rel=1e-12)
    assert inclusion_probability(20) > BOOTSTRAP_INCLUSION_LIMIT
    # it decreases monotonically to 1 - 1/e from above
    vals = [inclusion_probability(n) for n in (5, 20, 100, 10_000)]
    assert vals[0] > vals[1] > vals[2] > vals[3] > BOOTSTRAP_INCLUSION_LIMIT
    assert vals[-1] == pytest.approx(BOOTSTRAP_INCLUSION_LIMIT, abs=1e-4)


def test_err_boot_is_biased_low_relative_to_the_leave_one_out_bootstrap():
    """The section's central claim, measured. (7.54) tests on points
    it also trained on, so it must come out below (7.56), which does
    not -- and both must sit above the training error."""
    rng = np.random.default_rng(37)
    order = []
    for _ in range(8):
        X = rng.normal(size=(80, 3))
        y = X @ [1.0, -1.0, 0.5] + rng.normal(scale=0.5, size=80)
        o = esl_bootstrap_err(X, y, B=60, seed=int(rng.integers(1e6)))
        order.append((o["err_train"], o["err_boot"], o["err_loo_boot"]))
        assert o["optimistic"] is True
    tr, bt, lo = map(np.mean, zip(*order))
    assert tr < bt < lo


def test_err_boot_optimism_is_severe_when_the_rule_interpolates():
    """The book's argument in its sharpest form. A rule that
    memorises the training set has zero training error, and (7.54)
    inherits most of that: only the roughly 36.8% of (b, i) pairs
    where i is out of bag contribute anything. The leave-one-out
    bootstrap keeps only those pairs and recovers the honest number."""
    rng = np.random.default_rng(41)
    n = 60
    X = rng.normal(size=(n, 1))
    y = rng.normal(size=n)          # no signal at all: truth is Var(y)

    def one_nn(Xtr, ytr):
        def pred(Xn):
            d = np.abs(Xn[:, 0][:, None] - Xtr[:, 0][None, :])
            return ytr[np.argmin(d, axis=1)]
        return pred

    o = esl_bootstrap_err(X, y, model=one_nn, B=200, seed=3)
    truth = 2.0 * float(np.var(y))   # E(y_i - y_j)^2 for independent draws
    assert o["err_train"] == pytest.approx(0.0, abs=1e-12)
    assert o["err_boot"] < 0.6 * o["err_loo_boot"]
    assert o["err_loo_boot"] == pytest.approx(truth, rel=0.35)


def test_632_reproduces_the_books_worked_failure_and_632_plus_fixes_it():
    """ESL p.252 verbatim: for a 1-nearest-neighbour rule on two
    equal classes with labels independent of the inputs,
    err_bar = 0 and Err^(1) = 0.5, so Err^(.632) = .632 x 0.5 =
    0.316 while the true error rate is 0.5. With gamma = 0.5 the
    relative overfitting rate is 1, the weight is 1, and .632+
    returns 0.5."""
    o = esl_oob_632(0.0, 0.5, gamma=0.5)
    assert o["err_632"] == pytest.approx(0.316, abs=1e-12)
    assert o["relative_overfitting_rate"] == pytest.approx(1.0, rel=1e-12)
    assert o["weight"] == pytest.approx(1.0, rel=1e-12)
    assert o["err_632_plus"] == pytest.approx(0.5, rel=1e-12)


def test_632_weight_runs_from_632_to_1_and_brackets_the_estimate():
    """(7.61): w ranges from .632 at R = 0 to 1 at R = 1, so the
    .632+ estimate ranges from Err^(.632) to Err^(1) and never
    outside."""
    et, e1 = 0.2, 0.5
    plain = esl_oob_632(et, e1)["err_632"]
    seen = []
    for gam in (0.5, 0.7, 1.0, 3.0):
        o = esl_oob_632(et, e1, gamma=gam)
        assert 0.632 - 1e-12 <= o["weight"] <= 1 + 1e-12
        assert min(plain, e1) - 1e-12 <= o["err_632_plus"] <= max(plain, e1) + 1e-12
        seen.append((o["relative_overfitting_rate"], o["err_632_plus"]))
    # more overfitting (larger R) pulls the estimate toward Err^(1)
    seen.sort()
    assert seen[0][1] <= seen[-1][1]
    assert esl_oob_632(et, e1, gamma=1e9)["weight"] == pytest.approx(
        0.632, rel=1e-6)


def test_632_takes_the_leave_one_out_bootstrap_not_err_boot():
    """The distinction the section is about. Since err_boot < Err^(1)
    always, feeding err_boot in gives a strictly smaller answer -- a
    downward correction applied to a quantity that was already biased
    downward."""
    rng = np.random.default_rng(43)
    X = rng.normal(size=(80, 3))
    y = X @ [1.0, -1.0, 0.5] + rng.normal(scale=0.5, size=80)
    b = esl_bootstrap_err(X, y, B=80, seed=5)
    right = esl_oob_632(b["err_train"], b["err_loo_boot"])["err_632"]
    wrong = esl_oob_632(b["err_train"], b["err_boot"])["err_632"]
    assert wrong < right
    assert "biased DOWNWARD" in b["optimism_note"]


def test_632_gamma_from_the_double_sum_and_the_dichotomous_formula_agree():
    """(7.58) and (7.59) are the same quantity. For 0/1 data the
    double sum of squared error over all (i, i') pairs equals
    p1(1-q1) + (1-p1)q1 exactly."""
    rng = np.random.default_rng(47)
    y = (rng.random(200) < 0.3).astype(float)
    yhat = (rng.random(200) < 0.45).astype(float)
    p1, q1 = float(y.mean()), float(yhat.mean())
    a = esl_oob_632(0.1, 0.2, y=y, y_pred=yhat)["gamma"]
    b = esl_oob_632(0.1, 0.2, p1=p1, q1=q1)["gamma"]
    assert a == pytest.approx(b, rel=1e-12)
    assert a == pytest.approx(p1 * (1 - q1) + (1 - p1) * q1, rel=1e-12)


def test_632_omits_the_plus_variant_when_gamma_is_unavailable():
    o = esl_oob_632(0.2, 0.4)
    assert o["err_632_plus"] is None
    assert o["value"] == o["err_632"] == pytest.approx(0.368 * 0.2 + 0.632 * 0.4)


# ---------------------------------------------------------- Ch. 15


def test_random_forest_uses_the_regression_mtry_rule():
    """floor(p/3) for regression, floor(sqrt(p)) for classification.
    They are not interchangeable, and which is larger flips at
    p = 9, where both equal 3: below it the classification rule is
    larger, above it the regression rule is, by a widening margin."""
    assert default_mtry(9) == 3 and default_mtry(9, "classification") == 3
    for p in (2, 4, 8):
        assert default_mtry(p, "classification") >= default_mtry(p)
    for p in (12, 30, 100):
        assert default_mtry(p) > default_mtry(p, "classification")
    assert default_mtry(100) == 33 and default_mtry(100, "classification") == 10
    assert default_mtry(2) == 1                 # never below one
    with pytest.raises(ValueError, match="regression"):
        default_mtry(5, "clustering")
    rng = np.random.default_rng(53)
    X = rng.normal(size=(120, 6))
    y = X[:, 0] + rng.normal(scale=0.3, size=120)
    assert esl_random_forest(X, y, B=20)["mtry"] == 2


def test_random_forest_out_of_bag_error_exceeds_training_error():
    """OOB predictions come only from trees that did not see the
    observation, so they must be worse than the in-bag fit. If they
    are not, the out-of-bag bookkeeping is wrong."""
    rng = np.random.default_rng(59)
    n = 250
    X = rng.normal(size=(n, 5))
    y = np.sin(2 * X[:, 0]) + X[:, 1] ** 2 - X[:, 2] + rng.normal(scale=0.3,
                                                                  size=n)
    o = esl_random_forest(X, y, B=60)
    assert o["oob_mse"] > o["train_mse"]
    assert o["n_oob_missing"] == 0
    assert o["subset_drawn_per"] == "node"
    # it still learns: OOB error beats predicting the mean
    assert o["oob_mse"] < np.var(y)


def test_random_forest_averaging_reduces_variance():
    """Ch. 15's thesis: bagged trees are identically distributed, so
    averaging cannot change the bias -- the only gain is variance
    reduction, and it must show up as a more stable prediction as B
    grows."""
    rng = np.random.default_rng(61)
    n = 150
    X = rng.normal(size=(n, 4))
    y = X[:, 0] * 2 - X[:, 1] + rng.normal(scale=0.5, size=n)
    grid = rng.normal(size=(30, 4))
    spread = {}
    for B in (1, 40):
        preds = np.vstack([
            esl_random_forest(X, y, B=B, newdata=grid, seed=s)["prediction"]
            for s in range(6)])
        spread[B] = float(np.mean(np.var(preds, axis=0)))
    assert spread[40] < spread[1] / 3


def test_random_forest_is_reproducible_and_validates_its_inputs():
    rng = np.random.default_rng(67)
    X = rng.normal(size=(60, 4))
    y = X[:, 0] + rng.normal(scale=0.4, size=60)
    a = esl_random_forest(X, y, B=15, seed=7)["prediction"]
    b = esl_random_forest(X, y, B=15, seed=7)["prediction"]
    assert np.array_equal(a, b)
    assert not np.array_equal(a, esl_random_forest(X, y, B=15,
                                                   seed=8)["prediction"])
    with pytest.raises(ValueError, match="mtry must lie"):
        esl_random_forest(X, y, B=5, mtry=99)
    with pytest.raises(ValueError, match="at least one tree"):
        esl_random_forest(X, y, B=0)
    with pytest.raises(ValueError, match="columns"):
        esl_random_forest(X, y, B=5, newdata=rng.normal(size=(3, 2)))


def test_random_forest_predicts_new_points():
    """A forest that only reproduces its training rows is a lookup
    table. On a linear signal the prediction at unseen points must
    track the truth."""
    rng = np.random.default_rng(71)
    X = rng.normal(size=(400, 3))
    y = 2 * X[:, 0] - X[:, 1] + rng.normal(scale=0.2, size=400)
    Q = rng.normal(size=(60, 3))
    truth = 2 * Q[:, 0] - Q[:, 1]
    pred = esl_random_forest(X, y, B=80, mtry=3, newdata=Q)["prediction"]
    assert np.corrcoef(pred, truth)[0, 1] > 0.9
