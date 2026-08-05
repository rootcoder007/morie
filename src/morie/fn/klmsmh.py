# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""RTS backward recursion applied to an existing forward pass.

Rauch, Tung and Striebel (1965), AIAA Journal 3(8):1445-1450,
doi:10.2514/3.3166.  This is the backward half only: the filtered
means and covariances are supplied, so the smoother costs one sweep
and no re-filtering.  For t = n-1 down to 1,

    C_t = P_{t|t} F' P_{t+1|t}^{-1},
    x_{t|n} = x_{t|t} + C_t (x_{t+1|n} - x_{t+1|t}).

The last smoothed state always equals the last filtered state, which
is the identity that anchors the recursion.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["kalman_smoother"]


def _get(model, key):
    v = model.get(key) if hasattr(model, "get") else getattr(model, key, None)
    if v is None:
        raise ValueError("kalman_smoother: model is missing entry " + key)
    return v


def kalman_smoother(y, model, filtered, ridge=1e-12):
    """Backward pass over supplied filtered quantities.

    Parameters
    ----------
    y : observation matrix, used only for its length.
    model : object with entry F (and the rest of the system, unused here).
    filtered : object with entries state, cov, predicted, predicted_cov,
        exactly as returned by the Kalman filter.
    """
    Y = core.mat(y)
    n = len(Y)
    if n == 0:
        raise ValueError("kalman_smoother: y is empty")
    F = core.mat(_get(model, "F"))
    d = len(F)
    xs = [core.vec(v) for v in _get(filtered, "state")]
    Ps = [core.mat(v) for v in _get(filtered, "cov")]
    xp = [core.vec(v) for v in _get(filtered, "predicted")]
    Pp = [core.mat(v) for v in _get(filtered, "predicted_cov")]
    if not (len(xs) == len(Ps) == len(xp) == len(Pp) == n):
        raise ValueError("kalman_smoother: filtered quantities do not have n entries")
    xsm = [list(v) for v in xs]
    Psm = [[r[:] for r in M] for M in Ps]
    for t in range(n - 2, -1, -1):
        A = [[Pp[t + 1][i][j] + (float(ridge) if i == j else 0.0) for j in range(d)] for i in range(d)]
        PF = [[sum(Ps[t][i][k] * F[j][k] for k in range(d)) for j in range(d)] for i in range(d)]
        C = [core.cholsolve(A, PF[i]) for i in range(d)]
        dx = [xsm[t + 1][j] - xp[t + 1][j] for j in range(d)]
        xsm[t] = [xs[t][j] + sum(C[j][k] * dx[k] for k in range(d)) for j in range(d)]
        dP = [[Psm[t + 1][i][j] - Pp[t + 1][i][j] for j in range(d)] for i in range(d)]
        CdP = core.matmul(C, dP)
        Psm[t] = [[Ps[t][i][j] + sum(CdP[i][k] * C[j][k] for k in range(d)) for j in range(d)] for i in range(d)]
    return RichResult(
        title="RTS backward recursion",
        summary_lines=[("n", n), ("state dim", d)],
        payload={
            "estimate": xsm[0][0],
            "smoothed": xsm,
            "smoothed_cov": Psm,
            "n": n,
            "method": "backward RTS pass over supplied filtered quantities, Rauch, Tung & Striebel (1965)",
        },
    )


def cheatsheet():
    return "klmsmh: RTS backward recursion"
