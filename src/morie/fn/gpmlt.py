# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Multi-task Gaussian process.

Bonilla, Chai and Williams (2008), "Multi-task Gaussian process
prediction", NIPS 20, equations (1)-(2): the covariance between the
values of task l at x and task m at x' factorises as

    < f_l(x) f_m(x') > = K^f_{lm} k^x(x, x'),

so the joint covariance is the Kronecker product K^f (x) K^x.  A
diagonal K^f means the tasks share nothing and the model collapses to
independent single-task GPs -- the tests check that collapse exactly,
because it is the only way to be sure the Kronecker indexing is the
right way round.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["gp_multitask"]


def _k(A, B, ell, var):
    out = []
    for i in range(len(A)):
        row = []
        for j in range(len(B)):
            s = 0.0
            for c in range(len(A[i])):
                d = A[i][c] - B[j][c]
                s += d * d
            row.append(var * math.exp(-0.5 * s / (ell * ell)))
        out.append(row)
    return out


def gp_multitask(X, y_tasks, X_test=None, task_cov=None, lengthscale=1.0, variance=1.0, noise=0.01):
    """Joint GP over T tasks observed at the same inputs.

    Parameters
    ----------
    X : n x d inputs shared by every task.
    y_tasks : T x n matrix of observations, one row per task.
    task_cov : T x T inter-task covariance; the identity by default.
    """
    A = core.mat(X)
    n = len(A)
    if n == 0:
        raise ValueError("gp_multitask: X is empty")
    Y = core.mat(y_tasks)
    T = len(Y)
    if T == 0:
        raise ValueError("gp_multitask: no tasks supplied")
    for r in Y:
        if len(r) != n:
            raise ValueError("gp_multitask: every task needs one observation per input")
    Kf = [[1.0 if i == j else 0.0 for j in range(T)] for i in range(T)] if task_cov is None else core.mat(task_cov)
    if len(Kf) != T or len(Kf[0]) != T:
        raise ValueError("gp_multitask: task_cov must be T x T")
    ell = float(lengthscale)
    var = float(variance)
    s2 = float(noise)
    if ell <= 0 or var <= 0:
        raise ValueError("gp_multitask: lengthscale and variance must be positive")
    if s2 < 0:
        raise ValueError("gp_multitask: noise must be non-negative")
    Xs = A if X_test is None else core.mat(X_test)
    Kx = _k(A, A, ell, var)
    N = T * n
    K = [[0.0] * N for _ in range(N)]
    for a in range(T):
        for b in range(T):
            for i in range(n):
                for j in range(n):
                    K[a * n + i][b * n + j] = Kf[a][b] * Kx[i][j]
    for i in range(N):
        K[i][i] += s2
    yv = [Y[a][i] for a in range(T) for i in range(n)]
    alpha = core.cholsolve(K, yv)
    Ksx = _k(Xs, A, ell, var)
    mean = []
    for a in range(T):
        row = []
        for j in range(len(Xs)):
            s = 0.0
            for b in range(T):
                for i in range(n):
                    s += Kf[a][b] * Ksx[j][i] * alpha[b * n + i]
            row.append(s)
        mean.append(row)
    L = core.chol(K)
    ll = -0.5 * sum(yv[i] * alpha[i] for i in range(N)) - sum(math.log(L[i][i]) for i in range(N)) - 0.5 * N * math.log(2.0 * math.pi)
    return RichResult(
        title="Multi-task GP",
        summary_lines=[("tasks", T), ("n", n)],
        payload={
            "estimate": mean[0][0],
            "mean": mean,
            "loglik": ll,
            "tasks": T,
            "n": n,
            "method": "K = K^f (x) K^x, Bonilla, Chai & Williams (2008) eqs. (1)-(2)",
        },
    )


def cheatsheet():
    return "gpmlt: multi-task Gaussian process"


# compact alias per ledger/NAMING.md
gpmultitask = gp_multitask
