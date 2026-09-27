"""MARS (ESL 9.4) against a naive re-implementation of the same algorithm with full least squares."""

import math

from morie.fn.eslmrs import _eval_term, esl_mars
from morie.fn.linsys import _householder_ls


def data():
    i = list(range(1, 61))
    X = [[((7 * t) % 59) / 59 * 3, ((11 * t) % 61) / 61 * 2] for t in i]
    y = [max(r[0] - 1, 0) + 0.5 * max(1.2 - r[1], 0) + 0.05 * math.cos(9 * t) for r, t in zip(X, i)]
    return X, y


def rss_of(X, y, terms):
    try:
        return _householder_ls([[_eval_term(tm, r) for tm in terms] for r in X], y)[1]
    except ValueError:
        return math.inf


def test_forward_steps_are_greedy_optimal():
    X, y = data()
    r = esl_mars(X, y, max_terms=7)
    fw = r["forward_terms"]
    assert fw[1][0][:2] == (0, 0.864406779661017) and fw[3][0][:2] == (1, 1.278688524590164)  # earth's first two pairs
    # at each step, no single candidate hinge (with its mirror) gives a lower RSS than the chosen one
    k = 1
    while k < len(fw):
        base = fw[:k]
        chosen = [
            t
            for t in fw[k : k + 2]
            if t[:-1] == fw[k][:-1] and [(a, b) for a, b, _ in t] == [(a, b) for a, b, _ in fw[k]]
        ]
        best = min(
            rss_of(X, y, base + [base[m] + [(j, t, 1)], base[m] + [(j, t, -1)]])
            for m in range(len(base))
            if len(base[m]) < 1
            for j in range(2)
            for t in {rw[j] for rw in X}
        )
        assert rss_of(X, y, base + chosen) <= best + 1e-9
        k += len(chosen)


def test_backward_pass_and_gcv():
    X, y = data()
    r = esl_mars(X, y, query=[[0.5, 0.3], [2.5, 1.8]])
    N = 60
    M = len(r["terms"]) + 2 * (len(r["terms"]) - 1) / 2
    assert abs(r["gcv"] - r["rss"] / N / (1 - M / N) ** 2) < 1e-15
    assert abs(r["rss"] - rss_of(X, y, r["terms"])) < 1e-12
    assert all(
        abs(a - sum(c * _eval_term(tm, q) for c, tm in zip(r["coefficients"], r["terms"]))) < 1e-12
        for a, q in zip(r["prediction"], [[0.5, 0.3], [2.5, 1.8]])
    )
    assert r["gcv"] < 0.00206728103486534  # lower than earth's GCV on the same data
