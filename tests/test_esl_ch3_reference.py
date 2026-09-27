"""ESL chapter 3 functions against R: lm, anova, closed-form ridge, glmnet, lars, pls::pcr."""

import math

from morie.fn import (
    esl_f_test,
    esl_lasso,
    esl_least_angle_reg,
    esl_ols_normal_equations,
    esl_pcr,
    esl_ridge,
    esl_z_score,
)


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol * max(1.0, abs(b))


def data():
    i = range(1, 31)
    X = [[math.sin(t), math.cos(2 * t) + 0.3 * math.sin(t), math.log(t), ((13 * t) % 11) / 11] for t in i]
    y = [1 + 2 * r[0] - r[1] + 0.5 * r[2] + 0.3 * math.cos(5 * t) for r, t in zip(X, i)]
    return X, y


def test_ols_z_f_equal_lm():
    X, y = data()
    Xi = [[1.0] + r for r in X]
    o = esl_ols_normal_equations(Xi, y)
    lm_b = (0.84643335323195024, 1.90920986972890305, -0.99909489266932239, 0.50613805193725347, 0.31549314097731329)
    lm_se = (
        0.130812544128734359,
        0.057959731581011885,
        0.053591693479459405,
        0.045855756038257429,
        0.139538640601853031,
    )
    assert all(close(a, b) for a, b in zip(o["beta"], lm_b)) and all(close(a, b) for a, b in zip(o["se"], lm_se))
    z = esl_z_score(Xi, y, o["beta"])
    for a, b in zip(
        z["p_t"],
        (
            8.9150333417878884e-07,
            4.0795635409506644e-22,
            3.5236322670360925e-16,
            4.2171624440190065e-11,
            3.2716313129758784e-02,
        ),
    ):
        assert close(a, b, 1e-9)
    f = esl_f_test([0, 1], [0, 1, 2, 3], X, y)  # anova(lm(y ~ X[, 1:2]), lm(y ~ X))
    assert close(f["statistic"], 66.262923419164778) and close(f["p_value"], 1.0170734578483357e-10, 1e-9)


def test_ridge_closed_form():
    # (X'X + 2 diag(0, 1, 1, 1, 1))^-1 X'y; this failed until the array core paired boolean masks as numpy does
    X, y = data()
    r = esl_ridge([[1.0] + v for v in X], y, 2.0)
    for a, b in zip(
        r["beta"],
        (1.07736105237374846, 1.63021964120020924, -0.81824940637693333, 0.42407348656085098, 0.27534255607623237),
    ):
        assert close(a, b)
    assert close(r["effective_df"], 4.1839958027358204) and close(r["rss"], 2.3519925493032678)


def test_lasso_lar_pcr():
    X, y = data()
    b = esl_lasso([[1.0] + v for v in X], y, 1.5)["beta"]  # glmnet(X, y, lambda = 1.5 / 30, standardize = FALSE)
    for a, c in zip(b, (1.20771080274177578, 1.78698515947479208, -0.87307341356985324, 0.42033601326284487, 0.0)):
        assert close(a, c, 1e-10)
    path = esl_least_angle_reg(X, y)["coef_path"].tolist()  # lars(type = "lar")
    for a, c in zip(path[3], (1.57885647331838275, -0.69321412555539508, 0.27324335738453653, 0.0)):
        assert close(a, c)
    p = esl_pcr(X, y, 2)  # pls::pcr(ncomp = 2, scale = FALSE)
    for a, c in zip(
        p["beta"] + [p["intercept"]],
        (0.300266243113946452, 0.366667070497064096, 0.021281516408693080, 0.025330623817156995, 2.244155835906887386),
    ):
        assert close(a, c)


def test_boolean_mask_pairs():
    from morie.fn import _array_core as np

    P = np.arange(25.0).reshape(5, 5)
    m = np.array([True, False, True, False, False])
    assert P[m, m].tolist() == [0.0, 12.0]  # numpy pairs the nonzero() positions
    E = np.eye(3)
    E[np.array([True, False, False]), np.array([True, False, False])] = 0.0
    assert E.tolist() == [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
