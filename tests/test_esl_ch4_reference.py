"""ESL chapter 4 against R: MASS::lda/qda formulas, glm, lm, nnet::multinom, glmnet."""

import math

from morie.fn import esl_iwls, esl_lda_disc, esl_logistic_reg, esl_qda
from morie.fn.eslind import esl_indicator_regression
from morie.fn.esll1l import esl_l1_logistic
from morie.fn.eslmnl import esl_multinomial_logit
from morie.fn.eslmol import esl_multi_output_ls


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol * max(1.0, abs(b))


def data():
    i = list(range(1, 41))
    X = [[math.sin(t) + 0.1 * t / 10, math.cos(3 * t) + (t % 3) * 0.8] for t in i]
    yb = [1 if (math.sin(3 * t) + r[0] + 0.4 * r[1]) > 0.5 else 0 for r, t in zip(X, i)]
    return i, X, yb, [[0.2, 0.5], [-0.5, 1.9]]


def test_logistic_and_iwls_equal_glm():
    i, X, yb, _ = data()
    # glm(yb ~ X, binomial, control = glm.control(epsilon = 1e-15))
    for f in (esl_logistic_reg, esl_iwls):
        r = f(X, yb)
        for a, b in zip(r["beta"], (-1.7658949316747921, 2.6256624770703652, 1.2896060481244269)):
            assert close(a, b, 1e-10)
        for a, b in zip(r["se"], (0.77633538360242160, 0.79847526366776544, 0.56745718134699019)):
            assert close(a, b, 1e-8)


def test_lda_qda_discriminants():
    i, X, _, Q = data()
    g = [t % 3 for t in i]
    a = esl_lda_disc(X, g, Q)
    for u, v in zip(
        a["discriminants"],
        (
            -1.1710168903230160,
            -0.85303945435103545,
            -2.2368809300709271,
            -1.6901568327291563,
            1.10915113996086401,
            2.2635147568254093,
        ),
    ):
        assert close(u, v)
    b = esl_qda(X, g, Q)
    for u, v in zip(
        b["discriminants"],
        (
            -0.90784632592712999,
            -0.20829140007268876,
            -2.10592636219517715,
            -6.85752565013955362,
            -1.99902648735511757,
            -0.95671668016566891,
        ),
    ):
        assert close(u, v)
    assert a["prediction"] == [1, 2] and b["prediction"] == [1, 2]  # MASS::lda / qda


def test_multi_output_and_indicator():
    i, X, _, Q = data()
    Y = [[r[0] + 0.5 * r[1] + 0.2 * math.cos(5 * t), r[1] - r[0] + 0.3 * math.sin(7 * t)] for r, t in zip(X, i)]
    m = esl_multi_output_ls(X, Y)  # lm(Y ~ X)
    ref = [
        [0.0056952417567327045, 0.025672468422435715],
        [0.9949320467203073282, -1.027295958522476127],
        [0.4965230837712901191, 0.998750812958504652],
    ]
    assert all(close(m["coefficients"][j][k], ref[j][k]) for j in range(3) for k in range(2))
    assert close(m["residual_covariance"][0][1], -0.0024688570690693094)
    r = esl_indicator_regression(X, [(t * 7) % 3 for t in i], Q)
    ref = [
        [0.4144355981502455544, 0.34969052374050963, 0.23587387810924471],
        [-0.0037528655142733802, 0.34336972205541372, 0.66038314345885962],
    ]
    assert all(close(r["fitted"][a][k], ref[a][k]) for a in range(2) for k in range(3))
    assert r["prediction"] == [0, 2]


def test_multinomial_and_l1_logistic():
    i, X, yb, _ = data()
    m = esl_multinomial_logit(X, [(t * 7) % 3 for t in i])  # multinom with baseline class 2
    ref = [
        [2.9242944380902962, 0.005467889849221046, -3.6695368955215137],
        [2.4176999205822489, -0.191029152082225651, -1.8377786718215035],
    ]
    assert m["converged"] and all(close(m["coefficients"][k][j], ref[k][j], 1e-8) for k in range(2) for j in range(3))
    assert close(m["se"][0][0], 1.1408228105325795, 1e-8) and close(m["loglik"], -29.753547780862426)
    r = esl_l1_logistic(X, yb, 2.0)  # glmnet(family = "binomial", lambda = 2 / 40, standardize = FALSE)
    for a, b in zip([r["intercept"]] + r["beta"], (-0.89013527368866274, 1.53994694644996089, 0.60509690422156026)):
        assert close(a, b, 1e-10)
    assert all(close(abs(s), 2.0, 1e-10) for s in r["score"])  # KKT, eq 4.32
