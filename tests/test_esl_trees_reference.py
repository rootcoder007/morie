"""ESL trees and boosting against R: rpart (anova, cp = 0), gbm (gaussian, stumps, bag.fraction = 1), AdaBoost properties."""

import math

from morie.fn import esl_adaboost, esl_adaboost_predict, esl_decision_tree, esl_gbm, esl_gbm_predict, esl_tree_predict


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol * max(1.0, abs(b))


def data():
    i = list(range(1, 61))
    X = [[((7 * t) % 59) / 59 * 3, ((11 * t) % 61) / 61 * 2] for t in i]
    y = [math.sin(2 * r[0]) + (1.0 if r[1] > 1.1 else 0.0) + 0.1 * math.cos(9 * t) for r, t in zip(X, i)]
    return X, y, [[0.5, 0.3], [2.5, 1.8], [1.4, 1.5]]


def test_tree_equals_rpart():
    X, y, Q = data()
    m = esl_decision_tree(X, y, max_depth=2, min_leaf=5)
    t = m["tree"]
    assert t["feature"] == 0 and close(t["threshold"], 1.6016949152542375)  # rpart: x1 >= 1.601695
    assert close(t["left"]["threshold"], 1.098360655737705) and close(t["right"]["threshold"], 1.098360655737705)
    for a, b in zip(esl_tree_predict(m, Q), (0.661485167792790, 0.258584826280236, 1.568193582397838)):
        assert close(a, b, 1e-13)
    assert esl_tree_predict(m, Q) == esl_tree_predict(t, Q)


def test_gbm_equals_gbm_package():
    X, y, Q = data()
    m = esl_gbm(X, y, M=20, nu=0.1, max_depth=1, min_leaf=5)
    for a, b in zip(esl_gbm_predict(m, Q), (0.673313781462843, 0.206052062699831, 1.136199473261762)):
        assert close(a, b, 1e-13)


def test_adaboost_properties():
    X, y, _ = data()
    yc = [1 if v > 0.5 else -1 for v in y]
    a = esl_adaboost(X, yc, M=5)
    n = len(X)
    w = [1 / n] * n
    for s, alpha in zip(a["stumps"], a["alphas"]):
        pred = [s["sign"] if r[s["feature"]] <= s["threshold"] else -s["sign"] for r in X]
        err = sum(wi for wi, p, t in zip(w, pred, yc) if p != t)
        # the stump attains the smallest weighted error over all stumps (brute force)
        best = min(
            sum(wi for wi, r, t in zip(w, X, yc) if (sg if r[j] <= thr else -sg) != t)
            for j in range(2)
            for thr in {r[j] for r in X}
            for sg in (1, -1)
        )
        assert close(err, best) and close(alpha, math.log((1 - err) / err))
        w = [wi * math.exp(alpha * (p != t)) for wi, p, t in zip(w, pred, yc)]
        tot = sum(w)
        w = [wi / tot for wi in w]
        assert close(sum(wi for wi, p, t in zip(w, pred, yc) if p != t), 0.5)
    assert esl_adaboost_predict(a, X) == a["prediction"]
