"""L1-penalised multinomial (ESL 18.19) against glmnet(family = "multinomial", type.multinomial = "ungrouped")."""

import math

from morie.fn.eslmn1 import esl_multinomial_l1


def test_equals_glmnet():
    i = list(range(1, 61))
    X = [[math.sin(t), math.cos(3 * t), math.sin(2 * t) * 0.5 + 0.3 * math.cos(t)] for t in i]
    g = [0 if math.sin(t) + 0.5 * math.cos(3 * t) > 0.3 else (1 if math.cos(3 * t) > -0.2 else 2) for t in i]
    r = esl_multinomial_l1(X, g, 1.5)
    assert r["converged"]
    ref_b0 = (0.320216519805003, -0.0834412368021782, -0.236775283002824)
    ref_B = (
        (3.889190633280150, 0.0, 0.526428643446192),
        (-0.7182021506431548, 0.0, 0.0),
        (0.0, -3.220516508612658, -1.021363980104159),
    )
    assert all(abs(a - b) < 1e-9 for a, b in zip(r["intercepts"], ref_b0))
    assert all(abs(a - b) < 1e-9 for ra, rb in zip(r["coefficients"], ref_B) for a, b in zip(ra, rb))
    assert r["coefficients"][0][1] == 0.0 and r["coefficients"][1][2] == 0.0  # exact zeros from the lasso
