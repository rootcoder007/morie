"""Known-answer tests for MVSML chapters 1-4.

Anchors are the values printed in the book: Table 1.1 fits (p.15-16),
the Table 2.9 GBLUP/SNP-BLUP equivalence (p.53-55), the Table 2.13 PCA
standard deviations (p.65) and the caret output for the chapter-4
binary example (p.135).
"""
import math

from morie.fn import _gp_core as gp
from morie.fn.msm001 import mvsml_general_eq_1_1
from morie.fn.msm002 import mvsml_general_eq_1_2
from morie.fn.msm003 import mvsml_general_eq_1_3
from morie.fn.msm005 import mvsml_general_eq_1_4
from morie.fn.msm006 import mvsml_general_eq_1_5
from morie.fn.msm257 import mvsml_general_eq_1_222
from morie.fn.msm240 import mvsml_preprocessing_eq_2_1
from morie.fn.msm241 import mvsml_preprocessing_eq_2_2
from morie.fn.msm242 import mvsml_preprocessing_eq_2_3
from morie.fn.msm243 import mvsml_preprocessing_eq_2_4
from morie.fn.msm334 import mvsml_preprocessing_eq_2_22
from morie.fn.msm332 import mvsml_elements_lin_reg_eq_3_1
from morie.fn.msm258 import mvsml_elements_lin_reg_eq_3_5
from morie.fn.msm007 import mvsml_overfitting_resampling_eq_4_9
from morie.fn.msm008 import mvsml_overfitting_resampling_eq_4_10
from morie.fn.msm009 import mvsml_overfitting_resampling_eq_4_14

# Table 1.1 p.14: grain yield, five environments, three reps
TAB11 = [[7.476, 7.298, 7.414],
         [7.117, 6.878, 7.004],
         [6.136, 6.340, 6.288],
         [5.600, 5.564, 5.466],
         [5.780, 5.948, 5.881]]

# Table 2.9 p.54: eight lines in two environments, seven SNPs
TAB29_Y = [5.215, 4.998, 5.284, 5.157, 6.601, 5.735, 5.565, 5.829]
TAB29_M = [[1, 0, 0, 1, 2, 1, 0],
           [1, 1, 2, 1, 0, 0, 1],
           [0, 0, 1, 1, 1, 1, 2],
           [2, 0, 0, 2, 1, 1, 1],
           [0, 1, 1, 1, 1, 2, 2],
           [2, 1, 1, 1, 0, 0, 0],
           [1, 0, 1, 0, 1, 0, 1],
           [1, 0, 1, 0, 1, 1, 0]]
TAB29_X = [[1, 0], [1, 0], [1, 0], [1, 0],
           [0, 1], [0, 1], [0, 1], [0, 1]]

# Table 2.13 p.65: fifteen lines, five variables on different scales
TAB213 = [[5.49, 179.98, 64.6, 119.39, 43.86],
          [6.84, 181.1, 64.68, 121.67, 44.72],
          [6.75, 181.0, 64.38, 120.1, 45.36],
          [4.98, 180.41, 64.03, 120.47, 44.21],
          [8.36, 180.89, 66.13, 122.3, 45.42],
          [4.43, 179.74, 63.26, 119.88, 43.65],
          [6.67, 180.49, 64.83, 120.22, 44.0],
          [4.44, 177.94, 62.31, 118.46, 42.6],
          [5.62, 178.91, 63.41, 120.64, 44.07],
          [6.96, 179.93, 64.03, 119.99, 43.53],
          [5.91, 181.21, 64.03, 120.86, 43.82],
          [5.59, 180.32, 64.89, 120.98, 45.21],
          [5.27, 180.97, 63.75, 120.14, 44.29],
          [4.32, 177.79, 62.47, 118.72, 42.28],
          [5.48, 180.04, 63.82, 120.08, 43.86]]

# p.134 binary example
Y_BIN = [1, 0, 1, 1, 0, 0, 1, 1, 0, 1]
YHAT_BIN = [1, 1, 1, 1, 0, 0, 1, 0, 0, 1]
PI_BIN = [0.6, 0.55, 0.8, 0.78, 0.3, 0.42, 0.9, 0.45, 0.3, 0.88]


def test_eq_1_1_systematic_plus_error():
    r = mvsml_general_eq_1_1([1.0, 2.0, 3.0], f=lambda v: 2 * v,
                             noise=[0.1, -0.1, 0.0])
    assert r["systematic"] == [2.0, 4.0, 6.0]
    assert abs(r["y"][0] - 2.1) < 1e-12
    assert abs(r["mean_error"]) < 1e-12


def test_eq_1_2_matches_book_single_mean_fit():
    r = mvsml_general_eq_1_2(TAB11)
    assert abs(r["beta"] - 6.4127) < 5e-5          # book p.15
    assert abs(r["sd_residual"] - 0.7197) < 5e-5   # book p.15


def test_eq_1_3_matches_book_environment_effects():
    r = mvsml_general_eq_1_3(TAB11)
    book = [7.396, 6.999, 6.255, 5.543, 5.869]     # book p.16
    for got, want in zip(r["beta"], book):
        assert abs(got - want) < 1e-3
    assert abs(r["sd_residual"] - 0.095) < 1e-3    # book p.16


def test_eq_1_4_deviations_sum_to_zero():
    r = mvsml_general_eq_1_4(TAB11)
    assert abs(r["deviations_sum"]) < 1e-12
    assert abs(r["beta_bar"] - 6.4127) < 5e-5


def test_eq_1_5_matches_book_variance_components():
    r = mvsml_general_eq_1_5(TAB11)
    assert abs(r["beta"] - 6.413) < 1e-3           # book p.16
    assert abs(r["sigma2_b"] - 0.594) < 1e-3       # book p.16
    assert abs(r["sd_residual"] - 0.095) < 1e-3
    assert abs(r["icc"] - 0.594 / (0.594 + 0.095 ** 2)) < 0.01


def test_model_comparison_shows_75x_drop():
    r = mvsml_general_eq_1_222(TAB11)
    # book: "residual standard error ... 7.47 times smaller"
    assert abs(r["estimate"] - 7.47) < 0.1


def test_eq_2_1_and_2_2_agree():
    # two-level random effect, identity residual covariance
    X = [[1.0], [1.0], [1.0], [1.0]]
    Z = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    y = [5.0, 5.4, 6.2, 6.6]
    Sigma = [[0.5, 0.0], [0.0, 0.5]]
    Sigma_inv = [[2.0, 0.0], [0.0, 2.0]]
    v = mvsml_preprocessing_eq_2_1(X, Z, y, Sigma)
    m = mvsml_preprocessing_eq_2_2(X, Z, y, Sigma_inv)
    assert abs(v["blue"][0] - m["blue"][0]) < 1e-9
    for a, b in zip(v["blup"], m["blup"]):
        assert abs(a - b) < 1e-9
    # BLUPs shrink toward zero relative to the raw level deviations
    assert abs(m["blup"][0]) < abs(5.2 - m["blue"][0]) + 1e-12


def test_gblup_and_snp_blup_give_the_same_gebv():
    # book p.55: "both methods give exactly the same breeding value
    # estimates".  G = Ms Ms'/p and Sigma = sigma2_g G is the same
    # model as Z = Ms with Sigma = sigma2_M I when sigma2_g = p sigma2_M.
    Ms = gp.scale_columns(TAB29_M)
    G = gp.grm_vanraden_method3(TAB29_M)
    p = len(TAB29_M[0])
    sigma2_m = 0.05
    g = mvsml_preprocessing_eq_2_3(TAB29_X, TAB29_Y, G,
                                   sigma2_g=p * sigma2_m)
    s = mvsml_preprocessing_eq_2_4(TAB29_X, TAB29_Y, Ms,
                                   sigma2_m=sigma2_m)
    for a, b in zip(g["gebv"], s["gebv"]):
        assert abs(a - b) < 1e-8
    # GEBVs are centred within the model, as breeding values should be
    assert abs(sum(g["gebv"])) < 1e-6


def test_grm_method3_is_symmetric_and_scaled():
    G = gp.grm_vanraden_method3(TAB29_M)
    n = len(G)
    for i in range(n):
        for j in range(n):
            assert abs(G[i][j] - G[j][i]) < 1e-12
    # trace/n is close to one because the columns were standardized
    assert abs(sum(G[i][i] for i in range(n)) / n - 1.0) < 0.35


def test_pca_matches_book_standard_deviations():
    r = gp.pca_compress(TAB213)
    book = [2.0090648, 0.6469991, 0.4964878, 0.4356803, 0.3297472]
    for got, want in zip(r["sd_pc"], book):
        assert abs(got - want) < 1e-6              # book p.65
    assert abs(r["cum_variance"][1] - 0.8909) < 1e-3  # book p.66
    assert len(r["compressed"][0]) == 5


def test_epe_grows_when_eigenvalues_vanish():
    good = mvsml_preprocessing_eq_2_22(1.0, [1.0, 1.0], [4.0, 4.0])
    bad = mvsml_preprocessing_eq_2_22(1.0, [1.0, 1.0], [4.0, 0.001])
    assert abs(good["estimate"] - 1.5) < 1e-12
    assert bad["estimate"] > 1000.0
    assert good["irreducible"] == 1.0


def test_ols_recovers_known_coefficients():
    X = [[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 6.0]]
    y = [1.0 + 2.0 * a + 3.0 * b for a, b in X]
    r = mvsml_elements_lin_reg_eq_3_1(X, y)
    assert abs(r["beta"][0] - 1.0) < 1e-8
    assert abs(r["beta"][1] - 2.0) < 1e-8
    assert abs(r["beta"][2] - 3.0) < 1e-8
    assert r["sigma2"] < 1e-16


def test_ridge_shrinks_and_reduces_to_ols():
    X = [[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 6.0]]
    y = [1.0 + 2.0 * a + 3.0 * b for a, b in X]
    zero = mvsml_elements_lin_reg_eq_3_5(X, y, lam=0.0)
    assert abs(zero["beta"][1] - 2.0) < 1e-6
    big = mvsml_elements_lin_reg_eq_3_5(X, y, lam=100.0)
    assert abs(big["beta"][1]) < abs(zero["beta"][1])
    assert big["prss"] > big["rss"] - 1e-12
    # the intercept is never penalized
    assert big["penalty"] == 100.0 * (big["beta"][1] ** 2
                                      + big["beta"][2] ** 2)


def test_binary_metrics_match_caret_output():
    # book p.135: positive class 0
    m = gp.binary_metrics(Y_BIN, YHAT_BIN, positive=0)
    assert abs(m["pccc"] - 0.8) < 1e-12
    assert abs(m["kappa"] - 0.5833) < 5e-5
    assert abs(m["sensitivity"] - 0.75) < 1e-12
    assert abs(m["specificity"] - 0.8333) < 5e-5
    assert abs(m["precision"] - 0.75) < 1e-12
    assert abs(m["neg_pred_value"] - 0.8333) < 5e-5
    assert abs(m["prevalence"] - 0.4) < 1e-12
    assert abs(m["detection_rate"] - 0.3) < 1e-12
    assert abs(m["balanced_accuracy"] - 0.7917) < 5e-5


def test_eq_4_9_and_4_10_one_versus_all():
    p = mvsml_overfitting_resampling_eq_4_9(Y_BIN, YHAT_BIN,
                                            class_index=0)
    s = mvsml_overfitting_resampling_eq_4_10(Y_BIN, YHAT_BIN,
                                             class_index=0)
    # TTP_all = 3 + 5 = 8, TFP_0 = 1, TFN_0 = 1
    assert p["TTP_all"] == 8 and p["TFP"] == 1
    assert abs(p["estimate"] - 8.0 / 9.0) < 1e-12
    assert s["TFN"] == 1
    assert abs(s["estimate"] - 8.0 / 9.0) < 1e-12
    assert abs(p["pCCC"] - 0.8) < 1e-12


def test_matthews_and_brier_and_mll():
    mcc = gp.matthews_corrcoef(Y_BIN, YHAT_BIN)
    # tp=5 tn=3 fp=1 fn=1 -> (15-1)/sqrt(6*6*4*4) = 14/24
    assert abs(mcc - 14.0 / 24.0) < 1e-12
    probs = [[1.0 - p, p] for p in PI_BIN]
    b = mvsml_overfitting_resampling_eq_4_14(probs, Y_BIN)
    hand = sum((p - y) ** 2 + ((1 - p) - (1 - y)) ** 2
               for p, y in zip(PI_BIN, Y_BIN)) / len(Y_BIN)
    assert abs(b["brier"] - hand) < 1e-12
    half = mvsml_overfitting_resampling_eq_4_14(probs, Y_BIN,
                                                halved=True)
    assert abs(half["brier"] - b["brier"] / 2.0) < 1e-12
    mll_hand = -sum(math.log(p if y == 1 else 1 - p)
                    for p, y in zip(PI_BIN, Y_BIN)) / len(Y_BIN)
    assert abs(b["mean_log_loss"] - mll_hand) < 1e-12


def test_gradient_descent_matches_ols():
    X = [[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 6.5]]
    y = [1.0 + 2.0 * a + 3.0 * b for a, b in X]
    ols = gp.ols_fit(X, y, add_intercept=True)
    gd = gp.gradient_descent_ols(X, y, optimal_step=True,
                                 add_intercept=True, tol=1e-12)
    for a, b in zip(gd["beta"], ols["beta"]):
        assert abs(a - b) < 1e-4
    assert gd["iterations"] > 1


def test_rank_deficient_designs_solve_via_pinv():
    """Regression: rank-deficient blocks used to raise a bare
    "singular matrix" from the linear-algebra core.  Both cases below
    are legitimate models, not user errors."""
    # (a) an all-zero covariate column
    r = gp.gxe_blup_model([5.0, 6.0, 5.4, 6.8], [[0.0]] * 4,
                          [[1, 0], [0, 1], [1, 0], [0, 1]],
                          [[1, 0, 0, 0], [0, 1, 0, 0],
                           [0, 0, 1, 0], [0, 0, 0, 1]],
                          [[1.0, 0.0], [0.0, 1.0]], 0.5,
                          [[0.3, 0.0], [0.0, 0.3]])
    assert len(r["b_lines"]) == 2
    assert r["b_lines"][1] > r["b_lines"][0]
    # (b) the literal eq (2.3) MME with a singular G (8 lines, 7 markers)
    G = gp.grm_vanraden_method3(TAB29_M)
    beta, u = gp.gblup_gebv(TAB29_X, TAB29_Y, G, 7 * 0.05,
                            use_mme=True)
    assert len(u) == 8
    assert all(v == v for v in u)          # no NaNs


def test_canonical_alias_for_the_mislabelled_stub():
    from morie.fn.msm334 import (
        mvsml_elements_lin_reg_expected_prediction_error as epe)
    assert epe(1.0, [1.0, 1.0], [4.0, 4.0])["estimate"] == \
        mvsml_preprocessing_eq_2_22(1.0, [1.0, 1.0],
                                    [4.0, 4.0])["estimate"]
