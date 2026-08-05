# morie.fn -- function file (rootcoder007/morie)
"""Extended Kalman filter."""

import math

from ._richresult import RichResult

__all__ = ["extended_kalman"]


def _mat(A):
    if isinstance(A, (int, float)):
        return [[float(A)]]
    out = []
    for r in A:
        if isinstance(r, (list, tuple)):
            out.append([float(v) for v in r])
        else:
            out.append([float(r)])
    return out


def _vec(v):
    if isinstance(v, (int, float)):
        return [float(v)]
    return [float(x) for x in v]


def _apply(g, x):
    return g(list(x)) if callable(g) else g


def extended_kalman(y, f, h, F, H, Q, R, x0=None, P0=None):
    """
    Extended Kalman filter

    Formula: linearize f, h via Jacobians at the current state, then run
    the linear Kalman recursions on the linearisation.

        predict   x- = f(x),        P- = F P F' + Q
        gain      S  = H P- H' + R, K  = P- H' / S
        update    x  = x- + K (y - h(x-)),  P = P- - K S K'

    Scalar observations, d-dimensional state.  ``f`` and ``h`` are the
    (possibly nonlinear) transition and observation maps; ``F`` and ``H``
    are their Jacobians, supplied either as callables of the state or as
    constant matrices (in which case the filter degenerates to the plain
    linear Kalman filter).

    Parameters
    ----------
    y : array-like
        Observation sequence (scalars).
    f : callable or array-like
        State transition x_{t} = f(x_{t-1}).
    h : callable or array-like
        Observation map y_t = h(x_t); returns a scalar.
    F : callable or array-like
        d x d Jacobian df/dx evaluated at the state.
    H : callable or array-like
        length-d Jacobian dh/dx evaluated at the state.
    Q : array-like
        d x d state noise covariance.
    R : float
        Observation noise variance (> 0).
    x0 : array-like, optional
        Initial state mean (default zeros).
    P0 : array-like, optional
        Initial state covariance (default identity).

    Returns
    -------
    result : dict
        Keys: estimate (final filtered first state component), state,
        cov, loglik, n, method.

    References
    ----------
    Schmidt (1966), in Leondes (ed.), Advances in Control Systems 3:293-340.
    Jazwinski (1970), Stochastic Processes and Filtering Theory, Academic Press.
    """
    y = _vec(y)
    n = len(y)
    if n == 0:
        raise ValueError("empty input: y has no observations")
    Q = _mat(Q)
    d = len(Q)
    if any(len(r) != d for r in Q):
        raise ValueError("Q must be square")
    R = float(R)
    if R <= 0.0:
        raise ValueError("R must be positive")
    x = [0.0] * d if x0 is None else _vec(x0)
    if len(x) != d:
        raise ValueError("x0 length must match Q")
    if P0 is None:
        P = [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]
    else:
        P = _mat(P0)
        if len(P) != d or any(len(r) != d for r in P):
            raise ValueError("P0 must be d x d")
    loglik = 0.0
    for t in range(n):
        xp = _vec(_apply(f, x))
        Fk = _mat(_apply(F, x))
        # P- = F P F' + Q
        FP = [[sum(Fk[i][k] * P[k][j] for k in range(d)) for j in range(d)]
              for i in range(d)]
        Pp = [[sum(FP[i][k] * Fk[j][k] for k in range(d)) + Q[i][j]
               for j in range(d)] for i in range(d)]
        hx = float(_apply(h, xp))
        Hk = _vec(_apply(H, xp))
        if len(Hk) != d:
            raise ValueError("H must return a length-d row")
        PH = [sum(Pp[i][k] * Hk[k] for k in range(d)) for i in range(d)]
        S = sum(Hk[i] * PH[i] for i in range(d)) + R
        K = [PH[i] / S for i in range(d)]
        v = y[t] - hx
        x = [xp[i] + K[i] * v for i in range(d)]
        P = [[Pp[i][j] - K[i] * S * K[j] for j in range(d)] for i in range(d)]
        loglik += -0.5 * (math.log(2.0 * math.pi * S) + v * v / S)
    return RichResult(payload={
        "estimate": x[0],
        "state": x,
        "cov": [P[i][j] for i in range(d) for j in range(d)],
        "loglik": loglik,
        "n": n,
        "method": "Extended Kalman filter",
    })


def cheatsheet():
    return "ekfF: Extended Kalman filter"


# compact alias per ledger/NAMING.md
extendedkalman = extended_kalman
