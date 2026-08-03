"""Known-answer tests for MVSML ch13 (CNN), ch14 (functional
regression) and ch15 (random forest for counts)."""
import math

from morie.fn import _gp_core as gp
from morie.fn.msm259 import (mvsml_deep_learning_eq_13_1,
                             mvsml_cnn_convolve)
from morie.fn.msm260 import mvsml_deep_learning_eq_13_2
from morie.fn.msm264 import mvsml_convolutional_nn_eq_14_3
from morie.fn.msm267 import mvsml_convolutional_nn_eq_14_4
from morie.fn.msm268 import mvsml_convolutional_nn_eq_14_5
from morie.fn.msm269 import mvsml_convolutional_nn_eq_14_6
from morie.fn.msm270 import mvsml_convolutional_nn_eq_14_7
from morie.fn.msm272 import mvsml_convolutional_nn_eq_14_9
from morie.fn.msm323 import mvsml_functional_regression_eq_15_1
from morie.fn.msm325 import mvsml_functional_regression_eq_15_2
from morie.fn.msm327 import mvsml_functional_regression_eq_15_3
from morie.fn.msm329 import mvsml_functional_regression_eq_15_4


# ------------------------------------------------ chapter 13
def test_conv_output_size_matches_the_book_example():
    # p.551: a 256x256 image with a 7x7 filter gives 250x250
    assert gp.conv_output_size(256, 7) == 250
    assert gp.conv_output_size(256, 7, stride=2) == 125


def test_conv_parameter_count_matches_the_book():
    # p.551: 7*7*3 + 1 = 148 against 256*256*3 + 1 = 196,609
    assert gp.conv_parameter_count(7, 3) == 148
    assert 256 * 256 * 3 + 1 == 196609


def test_eq_13_1_is_the_dot_product_of_the_receptive_field():
    img = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    ker = [[1.0, 0.0], [0.0, 1.0]]
    r = mvsml_deep_learning_eq_13_1(img, ker, bias=0.5)
    # top-left patch [[1,2],[4,5]] . [[1,0],[0,1]] + 0.5 = 1+5+0.5
    assert abs(r["feature_map"][0][0] - 6.5) < 1e-12
    assert r["output_shape"] == (2, 2)


def test_weight_sharing_gives_translational_invariance():
    # the same feature at a different position gives the same response
    ker = [[1.0, -1.0]]
    a = gp.conv2d([[5.0, 1.0, 0.0, 0.0]], ker)
    b = gp.conv2d([[0.0, 0.0, 5.0, 1.0]], ker)
    assert abs(a[0][0] - b[0][2]) < 1e-12


def test_eq_13_2_applies_the_activation():
    img = [[1.0, -2.0], [-3.0, 4.0]]
    ker = [[1.0]]
    r = mvsml_deep_learning_eq_13_2(img, ker, activation="relu")
    assert r["activation_map"] == [[1.0, 0.0], [0.0, 4.0]]


# ------------------------------------------------ chapter 14
T = [j / 20.0 for j in range(21)]


def _curves(n=12, seed=4):
    """Curves spanning several harmonics, so the design matrix of
    eq. (14.9) has full column rank and beta is identified."""
    rng = gp.np.random.default_rng(seed)
    out, truth = [], []
    for _ in range(n):
        a = float(rng.normal(0, 1))
        b = float(rng.normal(0, 1))
        c = float(rng.normal(0, 1))
        d = float(rng.normal(0, 1))
        out.append([a + b * math.sin(2 * math.pi * t)
                    + c * math.cos(2 * math.pi * t)
                    + d * math.sin(4 * math.pi * t) for t in T])
        truth.append(a)
    return out, truth


def test_eq_14_7_coefficients_reproduce_the_curve():
    # a curve built from the basis is recovered exactly (eq. 14.6/14.7)
    c = [1.5, -0.5, 2.0]
    r = mvsml_convolutional_nn_eq_14_6(T, c, kind="polynomial")
    back = mvsml_convolutional_nn_eq_14_7(T, r["x_t"], L2=3,
                                          kind="polynomial")
    for a, b in zip(back["c"], c):
        assert abs(a - b) < 1e-8


def test_eq_14_8_basis_matrix_shape_and_first_column():
    Psi = gp.fda_basis_matrix(T, 5, kind="fourier")
    assert len(Psi) == len(T) and len(Psi[0]) == 5
    assert all(abs(row[0] - 1.0) < 1e-12 for row in Psi)


def test_eq_14_9_design_matrix_has_the_intercept_column():
    X, _ = _curves()
    r = mvsml_convolutional_nn_eq_14_9(T, X, L1=3, L2=5)
    assert all(abs(row[0] - 1.0) < 1e-12 for row in r["X_star"])
    assert len(r["X_star"][0]) == 4          # 1 + L1
    assert len(r["Q"]) == 3 and len(r["Q"][0]) == 5


def test_eq_14_3_and_14_4_recover_a_known_linear_functional():
    # y depends on the curves through the design of eq. (14.3), so the
    # ML fit of eq. (14.4) must recover the generating coefficients
    X, _ = _curves(n=25, seed=9)
    d = mvsml_convolutional_nn_eq_14_3(T, X, L1=3, L2=5)
    Xs = d["X_star"]
    # the design must have full column rank for beta to be identified
    XtX = gp._mm(gp._t(Xs), Xs)
    _, lam = gp.is_positive_semidefinite(XtX)
    assert min(lam) > 1e-8
    true_beta = [2.0, 1.5, -1.0, 0.5]
    y = [sum(tb * v for tb, v in zip(true_beta, row)) for row in Xs]
    f = mvsml_convolutional_nn_eq_14_4(T, X, y, L1=3, L2=5)
    for a, b in zip(f["beta"], true_beta):
        assert abs(a - b) < 1e-6
    assert f["sigma2"] < 1e-14               # exact fit


def test_eq_14_4_fits_exactly_even_when_beta_is_not_identified():
    # with a rank-deficient design many beta give the same fit, so the
    # identified quantity is the fitted vector, not the coefficients
    rng = gp.np.random.default_rng(1)
    X = [[float(rng.normal(0, 1))
          + float(rng.normal(0, 1)) * math.sin(2 * math.pi * t)
          for t in T] for _ in range(20)]
    d = mvsml_convolutional_nn_eq_14_3(T, X, L1=3, L2=5)
    y = [sum(row) for row in d["X_star"]]
    f = mvsml_convolutional_nn_eq_14_4(T, X, y, L1=3, L2=5)
    for a, b in zip(f["fitted"], y):
        assert abs(a - b) < 1e-6


def test_eq_14_5_sigma2_divides_by_n():
    X, _ = _curves(n=15, seed=3)
    rng = gp.np.random.default_rng(2)
    y = [float(rng.normal(0, 1)) for _ in range(15)]
    f = mvsml_convolutional_nn_eq_14_5(T, X, y, L1=3, L2=5)
    hand = sum(v * v for v in f["residuals"]) / 15
    assert abs(f["sigma2"] - hand) < 1e-12


def test_beta_function_and_bic_and_loocv():
    b = gp.fda_beta_function(T, [1.0, 0.0, 0.0], 3)
    assert all(abs(v - 1.0) < 1e-12 for v in b)   # constant basis
    assert abs(gp.fda_bic(-10.0, 3, 100)
               - (20.0 + 4 * math.log(100))) < 1e-12
    x = [math.sin(2 * math.pi * t) for t in T]
    cv_good = gp.fda_loocv(T, x, 5)
    cv_poor = gp.fda_loocv(T, x, 1)
    assert cv_good < cv_poor                 # p.583


# ------------------------------------------------ chapter 15
def test_eq_15_1_links_invert_correctly():
    r = mvsml_functional_regression_eq_15_1(math.log(3.0), 0.0)
    assert abs(r["mu"] - 3.0) < 1e-12
    assert abs(r["theta"] - 0.5) < 1e-12


def test_eq_15_2_loglik_matches_the_formula():
    ys = [1.0, 2.0, 3.0]
    mu = 2.0
    r = mvsml_functional_regression_eq_15_2(ys, mu=mu)
    hand = (-3 * math.log(1 - math.exp(-mu))
            + math.log(mu) * 6 - 3 * mu
            - sum(math.lgamma(v + 1) for v in ys))
    assert abs(r["loglik"] - hand) < 1e-12


def test_zero_truncated_mle_solves_the_score_equation():
    # p.652: mean(Y+) = mu / (1 - exp(-mu))
    ys = [1.0, 2.0, 2.0, 3.0, 1.0]
    mu = gp.zero_truncated_poisson_mle(ys)
    lhs = sum(ys) / len(ys)
    assert abs(mu / (1 - math.exp(-mu)) - lhs) < 1e-8
    # and it maximizes the likelihood
    base = gp.zero_truncated_poisson_loglik(ys, mu)
    for d in (-0.05, 0.05):
        assert gp.zero_truncated_poisson_loglik(ys, mu + d) <= base


def test_split_maximizes_the_summed_child_loglik():
    y = [1.0, 1.0, 2.0, 8.0, 9.0, 9.0]
    x = [0.0, 1.0, 2.0, 8.0, 9.0, 10.0]
    r = mvsml_functional_regression_eq_15_2(y, x=x)
    s = r["split"]
    assert s["threshold"] is not None
    assert 1.0 <= s["threshold"] <= 8.0      # separates the groups


def test_eq_15_3_and_15_4_predictions():
    # (15.3) is the ZAP mean
    mu, th = 2.0, 0.3
    r3 = mvsml_functional_regression_eq_15_3(th, mu)
    hand = (1 - th) * math.exp(-mu) / (1 - math.exp(-mu))
    assert abs(r3["y_hat"] - hand) < 1e-12
    # (15.4) thresholds at 0.5 instead
    hi = mvsml_functional_regression_eq_15_4(0.7, mu)
    lo = mvsml_functional_regression_eq_15_4(0.3, mu)
    assert hi["y_hat"] == 0.0 and hi["is_zero"] is True
    assert abs(lo["y_hat"] - mu) < 1e-12


def test_canonical_aliases():
    from morie.fn.msm327 import mvsml_zap_predict
    assert mvsml_cnn_convolve is mvsml_deep_learning_eq_13_1
    assert mvsml_zap_predict is mvsml_functional_regression_eq_15_3
