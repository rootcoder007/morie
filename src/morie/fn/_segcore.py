"""Shared pieces for the Massey-Denton segregation indices: count matrices, sums, sorted cumulations."""

import math


def ssum(it):
    s = 0.0
    for v in it:
        s += v
    return s


def counts(x):
    X = [[float(v) for v in r] for r in x]
    if not X or any(len(r) != len(X[0]) for r in X) or len(X[0]) < 2:
        raise ValueError("counts must be a units x groups matrix with at least two groups")
    if any(v < 0 for r in X for v in r):
        raise ValueError("counts must be non-negative")
    X = [r for r in X if ssum(r) > 0]  # empty units carry no information (as OasisR drops them)
    return X


def col_totals(X):
    return [ssum(r[k] for r in X) for k in range(len(X[0]))]


def row_totals(X):
    return [ssum(r) for r in X]


def matvec(M, v):
    return [ssum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def decay(d, beta):
    return [[math.exp(-beta * float(v)) for v in r] for r in d]
