"""Known-answer tests for MVSML chapter 9 (SVM), eq. (9.4)-(9.33).

Example 9.1 on p.350 gives a hand computation with an exact Q Q'
matrix, which anchors the dual objective directly.
"""
import math

from morie.fn import _gp_core as gp
from morie.fn.msm171 import (mvsml_ridge_lasso_elastic_eq_9_4,
                             mvsml_svm_hyperplane_side)
from morie.fn.msm173 import mvsml_ridge_lasso_elastic_eq_9_5
from morie.fn.msm202 import mvsml_ridge_lasso_elastic_eq_9_28
from morie.fn.msm203 import mvsml_ridge_lasso_elastic_eq_9_29
from morie.fn.msm204 import mvsml_ridge_lasso_elastic_eq_9_30
from morie.fn.msm210 import mvsml_ridge_lasso_elastic_eq_9_31
from morie.fn.msm212 import mvsml_ridge_lasso_elastic_eq_9_32
from morie.fn.msm213 import mvsml_ridge_lasso_elastic_eq_9_33

# Example 9.1, p.350
EX_X = [[0.5, 1.0], [-0.5, 1.0], [-0.5, -1.0]]
EX_Y = [-1.0, -1.0, 1.0]


def test_example_9_1_label_matrix_and_gram_match_the_book():
    Q = gp.svm_label_matrix(EX_X, EX_Y)
    assert Q == [[-0.5, -1.0], [0.5, -1.0], [-0.5, -1.0]]
    QQt = gp._mm(Q, gp._t(Q))
    book = [[1.25, 0.75, 1.25],
            [0.75, 1.25, 0.75],
            [1.25, 0.75, 1.25]]
    for i in range(3):
        for j in range(3):
            assert abs(QQt[i][j] - book[i][j]) < 1e-12


def test_eq_9_4_hyperplane_of_figure_9_2():
    # Fig. 9.2 p.340: the hyperplane 1 + 2 X1 + 3 X2 = 0
    r = mvsml_ridge_lasso_elastic_eq_9_4([[1.0, 1.0], [-1.0, -1.0]],
                                          1.0, [2.0, 3.0])
    assert r["side"] == [1, -1]
    # a point exactly on the hyperplane is not on the positive side
    on = mvsml_ridge_lasso_elastic_eq_9_4([[-0.5, 0.0]], 1.0,
                                           [2.0, 0.0])
    assert on["side"] == [-1]


def test_eq_9_5_decision_values_and_labels():
    r = mvsml_ridge_lasso_elastic_eq_9_5(EX_X, 0.0, [1.0, -1.0])
    hand = [0.5 - 1.0, -0.5 - 1.0, -0.5 + 1.0]
    for a, b in zip(r["f"], hand):
        assert abs(a - b) < 1e-12
    assert r["labels"] == [-1, -1, 1]
    # for a separable fit y_i f(x_i) > 0 for every observation (p.341)
    assert all(y * v > 0 for y, v in zip(EX_Y, r["f"]))


def test_eq_9_28_weights_are_a_combination_of_training_vectors():
    alpha = [0.0, 2.0, 2.0]
    r = mvsml_ridge_lasso_elastic_eq_9_28(alpha, EX_X, EX_Y)
    hand = [sum(alpha[i] * EX_Y[i] * EX_X[i][j] for i in range(3))
            for j in range(2)]
    for a, b in zip(r["beta"], hand):
        assert abs(a - b) < 1e-12
    # only the nonzero multipliers contribute (p.348)
    assert r["support_vectors"] == [1, 2]


def test_eq_9_29_balance_condition():
    ok = mvsml_ridge_lasso_elastic_eq_9_29([1.0, 1.0, 2.0], EX_Y)
    assert ok["satisfied"] is True            # -1 -1 +2 = 0
    bad = mvsml_ridge_lasso_elastic_eq_9_29([1.0, 0.0, 0.0], EX_Y)
    assert bad["satisfied"] is False


def test_eq_9_32_and_9_31_agree_at_a_feasible_point():
    # (9.31) collapses to (9.32) once (9.29) holds, because the
    # balance term vanishes and the cross term is -2x the norm term
    alpha = [1.0, 1.0, 2.0]
    a = mvsml_ridge_lasso_elastic_eq_9_31(alpha, EX_X, EX_Y,
                                           beta0=3.7)
    b = mvsml_ridge_lasso_elastic_eq_9_32(alpha, EX_X, EX_Y)
    assert abs(a["L"] - b["L"]) < 1e-12
    # and the objective is quadratic: hand value from Q Q'
    Q = gp.svm_label_matrix(EX_X, EX_Y)
    QQt = gp._mm(Q, gp._t(Q))
    hand = sum(alpha) - 0.5 * sum(
        alpha[i] * alpha[j] * QQt[i][j]
        for i in range(3) for j in range(3))
    assert abs(b["L"] - hand) < 1e-12


def test_eq_9_33_feasibility():
    good = mvsml_ridge_lasso_elastic_eq_9_33([1.0, 1.0, 2.0], EX_Y)
    assert good["feasible"] is True
    neg = mvsml_ridge_lasso_elastic_eq_9_33([-1.0, 1.0, 0.0], EX_Y)
    assert neg["nonnegative"] is False
    over = mvsml_ridge_lasso_elastic_eq_9_33([1.0, 1.0, 2.0], EX_Y,
                                              C=1.5)
    assert over["bounded"] is False


def test_fitted_dual_satisfies_the_kkt_conditions():
    r = mvsml_ridge_lasso_elastic_eq_9_32(None, EX_X, EX_Y,
                                           fit=True)
    feas = gp.svm_dual_constraints_ok(r["alpha"], EX_Y)
    assert feas["feasible"] is True
    k = mvsml_ridge_lasso_elastic_eq_9_30(r["alpha"], EX_X, EX_Y,
                                           r["beta0"], r["beta"],
                                           tol=1e-2)
    assert k["satisfied"] is True
    # every support vector sits on the margin y_i f(x_i) = 1 (p.348)
    for i in r["support_vectors"]:
        assert abs(k["margin_slack"][i]) < 1e-2


def test_fitted_svm_separates_the_example_data():
    r = mvsml_ridge_lasso_elastic_eq_9_32(None, EX_X, EX_Y,
                                           fit=True)
    pred = gp.svm_predict(EX_X, r["beta0"], r["beta"])
    assert pred == [int(v) for v in EX_Y]
    assert len(r["support_vectors"]) >= 1      # each class has one


def test_dual_uses_only_inner_products():
    # p.349: the dual depends on the data only through x_i . x_j, so
    # passing the Gram matrix must give an identical objective
    alpha = [1.0, 1.0, 2.0]
    G = gp._mm(gp._mat(EX_X), gp._t(gp._mat(EX_X)))
    a = gp.svm_dual_objective(alpha, EX_X, EX_Y)
    b = gp.svm_dual_objective(alpha, EX_X, EX_Y, K=G)
    assert abs(a - b) < 1e-12
    # and a kernel can replace them
    K = gp.kernel_matrix(EX_X, kernel="gaussian", gamma=0.5)
    c = gp.svm_dual_objective(alpha, EX_X, EX_Y, K=K)
    assert c != a


def test_canonical_aliases():
    assert mvsml_svm_hyperplane_side is \
        mvsml_ridge_lasso_elastic_eq_9_4
