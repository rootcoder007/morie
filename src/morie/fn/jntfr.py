# morie.fn -- wave2 slice x_2_01 (rootcoder007/morie)
"""Joint shared-frailty model for recurrent events and a terminal event.

Liu, Wolfe and Huang (2004), "Shared frailty models for recurrent
events and a terminal event", Biometrics 60(3):747-756,
doi:10.1111/j.0006-341X.2004.00225.x.  A single cluster-level frailty
w_i drives both processes,

    lambda_R(t | w_i) = w_i lambda_0R(t),
    lambda_T(t | w_i) = w_i^alpha lambda_0T(t),

with w_i ~ Gamma(1/theta, 1/theta), so E[w] = 1 and Var[w] = theta.
alpha is the association parameter: alpha = 0 makes the terminal event
independent of the recurrent process, alpha = 1 is the ordinary shared
frailty, and alpha < 0 means clusters with many recurrences die later.

The baselines are taken to be constant (exponential), which leaves
lambda_0R, lambda_0T, theta and alpha to estimate.  Because
w^alpha appears in the terminal hazard the frailty cannot be integrated
out in closed form for general alpha, so the cluster contribution is
integrated numerically: the frailty expectation is taken as a 32-point
midpoint rule on the probability scale, w_q = F^-1((q + 1/2)/32), with
F the Gamma(1/theta, 1/theta) distribution function.  The rule is
fixed, so both language
arms evaluate exactly the same integral.  Maximisation is a
fixed-length coordinate golden-section search, again identical in both
arms.

When the frailty variance is negligible the likelihood factorises and
the maximisers approach the closed forms lambda_R = sum N_i / sum A_i
and lambda_T = sum delta_i / sum T_i; that limit, and the fact that the
quadrature nodes average to E[w] = 1, are the test anchors.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core
from ._richresult import RichResult

__all__ = ["joint_frailty"]

_NQ = 16
_NODE_CACHE = {}


def _nodes(theta):
    """Generalised Gauss-Laguerre rule for the Gamma(1/th, 1/th) density.

    Golub and Welsch (1969), Mathematics of Computation 23(106):221-230,
    doi:10.1090/S0025-5718-69-99647-1: the nodes are the eigenvalues of
    the Jacobi matrix of the orthogonal polynomials for the weight
    x^alpha exp(-x), and the weights are the squared first components of
    the corresponding eigenvectors, which already sum to one.
    """
    hit = _NODE_CACHE.get(theta)
    if hit is not None:
        return hit
    k = 1.0 / theta
    al = k - 1.0
    J = [[0.0] * _NQ for _ in range(_NQ)]
    for i in range(_NQ):
        J[i][i] = 2.0 * i + al + 1.0
    for i in range(1, _NQ):
        b = math.sqrt(i * (i + al))
        J[i][i - 1] = b
        J[i - 1][i] = b
    vals, vecs = core.jacobi(J)
    xs = [vals[q] / k for q in range(_NQ)]
    ws = [vecs[0][q] * vecs[0][q] for q in range(_NQ)]
    out = (xs, ws)
    _NODE_CACHE[theta] = out
    if len(_NODE_CACHE) > 4096:
        _NODE_CACHE.clear()
    return out


def _loglik(lamR, lamT, theta, alpha, N, A, dl, T):
    xs, ws = _nodes(theta)
    tot = 0.0
    for i in range(len(N)):
        acc = 0.0
        for q in range(_NQ):
            w = xs[q]
            lp = N[i] * math.log(lamR * w) - lamR * w * A[i]
            wa = w ** alpha
            lp += dl[i] * math.log(lamT * wa) - lamT * wa * T[i]
            acc += ws[q] * math.exp(lp)
        if acc <= 0.0:
            acc = 1e-300
        tot += math.log(acc)
    return tot


def _golden(f, lo, hi, iters=40):
    g = 0.6180339887498949
    c = hi - g * (hi - lo)
    d = lo + g * (hi - lo)
    fc = f(c)
    fd = f(d)
    for _ in range(iters):
        if fc > fd:
            hi, d, fd = d, c, fc
            c = hi - g * (hi - lo)
            fc = f(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + g * (hi - lo)
            fd = f(d)
    return 0.5 * (lo + hi)


def joint_frailty(time, event, terminal, cluster, sweeps=4):
    """Fit the Liu-Wolfe-Huang joint frailty model with constant baselines.

    Parameters
    ----------
    time : array-like
        Follow-up time of each unit (cluster member).
    event : array-like
        Number of recurrent events observed for that unit.
    terminal : array-like
        Terminal event indicator, 0 or 1.
    cluster : array-like
        Cluster (subject) label.
    sweeps : int
        Coordinate-ascent sweeps.
    """
    tv = core.vec(time)
    n = len(tv)
    if n == 0:
        raise ValueError("joint_frailty: time is empty")
    ev = core.vec(event)
    te = core.vec(terminal)
    cl = core.vec(cluster)
    if len(ev) != n or len(te) != n or len(cl) != n:
        raise ValueError("joint_frailty: time, event, terminal and cluster have different lengths")
    for v in tv:
        if v <= 0:
            raise ValueError("joint_frailty: time must be positive")
    for v in te:
        if v not in (0.0, 1.0):
            raise ValueError("joint_frailty: terminal must be 0 or 1")
    for v in ev:
        if v < 0:
            raise ValueError("joint_frailty: event counts must be non-negative")
    labels = []
    for v in cl:
        if v not in labels:
            labels.append(v)
    labels.sort()
    g = len(labels)
    N = [0.0] * g
    A = [0.0] * g
    dl = [0.0] * g
    T = [0.0] * g
    for i in range(n):
        j = labels.index(cl[i])
        N[j] += ev[i]
        A[j] += tv[i]
        dl[j] += te[i]
        T[j] += tv[i]
    if sum(N) <= 0 or sum(dl) <= 0:
        raise ValueError("joint_frailty: need at least one recurrent and one terminal event")
    lamR = sum(N) / sum(A)
    lamT = sum(dl) / sum(T)
    theta = 0.5
    alpha = 1.0
    for _ in range(int(sweeps)):
        lamR = _golden(lambda v: _loglik(v, lamT, theta, alpha, N, A, dl, T), 1e-4, 10.0 * sum(N) / sum(A))
        lamT = _golden(lambda v: _loglik(lamR, v, theta, alpha, N, A, dl, T), 1e-4, 10.0 * sum(dl) / sum(T))
        theta = _golden(lambda v: _loglik(lamR, lamT, v, alpha, N, A, dl, T), 1e-3, 5.0)
        alpha = _golden(lambda v: _loglik(lamR, lamT, theta, v, N, A, dl, T), -3.0, 3.0)
    ll = _loglik(lamR, lamT, theta, alpha, N, A, dl, T)
    return RichResult(
        title="Joint frailty for recurrent and terminal events",
        summary_lines=[("clusters", g), ("alpha", alpha), ("theta", theta)],
        payload={
            "estimate": alpha,
            "alpha": alpha,
            "theta": theta,
            "lambda_r": lamR,
            "lambda_t": lamT,
            "loglik": ll,
            "n_clusters": float(g),
            "n_recurrent": sum(N),
            "n_terminal": sum(dl),
            "exposure": sum(A),
            "naive_lambda_r": sum(N) / sum(A),
            "naive_lambda_t": sum(dl) / sum(T),
            "n": n,
            "method": "lambda_R = w lambda_0R, lambda_T = w^alpha lambda_0T, w ~ Gamma(1/theta, 1/theta), Liu, Wolfe & Huang (2004)",
        },
    )


def cheatsheet():
    return "jntfr: Joint frailty for recurrent + terminal events"


# compact alias per ledger/NAMING.md
jointfrailty = joint_frailty
