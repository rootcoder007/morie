"""K-class gradient boosting (ESL Alg. 10.4): leaf_scale = 1 equals gbm multinomial; ESL's (K-1)/K default."""

import math

from morie.fn.eslgbk import esl_gbm_multiclass


def data():
    i = list(range(1, 91))
    return [[math.sin(t) + ((t % 3) == 0) * 1.5, math.cos(2 * t) + ((t % 3) == 1) * 1.2] for t in i], [t % 3 for t in i]


def test_gbm_multinomial_and_esl_scale():
    X, g = data()
    q = [[0.2, 0.5], [1.5, -0.8]]
    r = esl_gbm_multiclass(X, g, M=10, nu=0.1, max_depth=1, min_leaf=5, query=q, leaf_scale=1.0)
    ref = (
        (0.202093559225793, 0.3642531923899715, 0.433653248384236),
        (0.744347530615853, 0.0658350911294626, 0.189817378254685),
    )
    assert all(
        abs(a - b) < 1e-12 for pr, rr in zip(r["prob"], ref) for a, b in zip(pr, rr)
    )  # gbm(distribution = "multinomial")
    e = esl_gbm_multiclass(X, g, M=10, nu=0.1, max_depth=1, min_leaf=5, query=q)
    assert all(
        abs(a - b) < 1e-12
        for a, b in zip(e["prob"][0], (0.26033967778951156, 0.34264694036234983, 0.39701338184813856))
    )
    path = e["deviance_path"]
    assert all(b < a for a, b in zip(path, path[1:]))
