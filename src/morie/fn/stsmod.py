# morie.fn -- function file (rootcoder007/morie)
"""Linear Gaussian state-space model."""

from __future__ import annotations

import math

from ._richresult import RichResult

__all__ = ["state_space_model"]


def _m(A):
    A = A.tolist() if hasattr(A, "tolist") else A
    if isinstance(A, (int, float)):
        return [[float(A)]]
    return [[float(v) for v in (r if isinstance(r, (list, tuple)) else [r])] for r in A]


def _mm(A, B):
    return [[math.fsum(a * b for a, b in zip(r, c)) for c in zip(*B)] for r in A]


def _t(A):
    return [list(c) for c in zip(*A)]


def _add(A, B, s=1.0):
    return [[a + s * b for a, b in zip(r, q)] for r, q in zip(A, B)]


def _mv(A, v):
    return [math.fsum(a * b for a, b in zip(r, v)) for r in A]


def _inv(A):
    n = len(A)
    M = [list(r) + [1.0 if i == j else 0.0 for j in range(n)] for i, r in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        d = M[c][c]
        M[c] = [v / d for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0.0:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [r[n:] for r in M]


def _logdet(A):
    n = len(A)
    M = [list(r) for r in A]
    s = 0.0
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        s += math.log(abs(M[c][c]))
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return s


def state_space_model(y, Z, T, H, Q, R, a1=None, P1=None):
    r"""Linear Gaussian state-space model.

    ``y_t = Z a_t + e_t``, ``e_t ~ N(0, H)``; ``a_{t+1} = T a_t + R n_t``,
    ``n_t ~ N(0, Q)``, ``a_1 ~ N(a1, P1)`` (Durbin and Koopman 2012,
    ch. 4). The Kalman filter ``v_t = y_t - Z a_t``, ``F_t = Z P_t Z' + H``,
    ``K_t = T P_t Z' F_t^{-1}``, ``a_{t+1} = T a_t + K_t v_t``, ``P_{t+1} = T
    P_t (T - K_t Z)' + R Q R'`` gives the prediction-error decomposition of
    the log-likelihood ``-np/2 log 2 pi - 1/2 sum (log|F_t| + v_t' F_t^{-1}
    v_t)``, the filtered states ``a_t + P_t Z' F_t^{-1} v_t``, and the
    backward recursion ``r_{t-1} = Z'F_t^{-1}v_t + L_t' r_t``, ``L_t = T -
    K_t Z`` the smoothed states ``a_t + P_t r_{t-1}`` (their eq. 4.44). Defaults
    ``a1 = 0`` and ``P1 = 10^6 I`` (a large-variance approximation of a
    diffuse start).

    Parameters
    ----------
    y : array-like, shape (n,) or (n, p)
        Observations.
    Z, T, H, Q, R : array-like
        System matrices (scalars allowed for one-dimensional pieces).
    a1, P1 : array-like, optional
        Initial state mean and covariance.

    Returns
    -------
    RichResult
        ``loglik``, ``filtered``, ``smoothed``, ``predicted`` (``a_t``),
        ``innovations``, ``F`` (innovation variances, p = 1: numbers),
        ``estimate`` (= loglik).

    References
    ----------
    Durbin, J. and Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*, 2nd ed.
    Oxford University Press, ch. 4.

    Examples
    --------
    >>> r = state_space_model([1.1, 0.9, 1.4, 1.2, 1.6], 1, 1, 0.5, 0.2, 1, a1=[0.0], P1=[[10.0]])
    >>> round(r["loglik"], 10), [round(v[0], 10) for v in r["smoothed"][:2]]
    (-6.0390961966, [1.1035944129, 1.1271040663])
    """
    Yr = y.tolist() if hasattr(y, "tolist") else list(y)
    Y = [[float(v)] if not isinstance(v, (list, tuple)) else [float(u) for u in v] for v in Yr]
    Zm, Tm, Hm, Qm, Rm = _m(Z), _m(T), _m(H), _m(Q), _m(R)
    m = len(Tm)
    p = len(Y[0])
    if len(Zm) != p or len(Zm[0]) != m or len(Hm) != p or len(Rm) != m or len(Qm) != len(Rm[0]):
        raise ValueError("inconsistent system matrix dimensions")
    a = [0.0] * m if a1 is None else [float(v) for v in (a1.tolist() if hasattr(a1, "tolist") else a1)]
    P = [[1e6 if i == j else 0.0 for j in range(m)] for i in range(m)] if P1 is None else _m(P1)
    RQR = _mm(_mm(Rm, Qm), _t(Rm))
    ll = 0.0
    keep = []
    for yt in Y:
        v = [a_ - b_ for a_, b_ in zip(yt, _mv(Zm, a))]
        PZt = _mm(P, _t(Zm))
        F = _add(_mm(Zm, PZt), Hm)
        Fi = _inv(F)
        K = _mm(_mm(Tm, PZt), Fi)
        Fv = _mv(Fi, v)
        ll -= 0.5 * (p * math.log(2 * math.pi) + _logdet(F) + math.fsum(x * w for x, w in zip(v, Fv)))
        filt = [a_ + b_ for a_, b_ in zip(a, _mv(PZt, Fv))]
        L = _add(Tm, _mm(K, Zm), -1.0)
        keep.append((a, P, v, Fi, L, filt, F))
        a = [x + w for x, w in zip(_mv(Tm, a), _mv(K, v))]
        P = _add(_mm(_mm(Tm, P), _t(L)), RQR)
    r = [0.0] * m
    smooth = [None] * len(Y)
    Zt = _t(Zm)
    for k in range(len(Y) - 1, -1, -1):
        at, Pt, v, Fi, L, _f, _F = keep[k]
        r = [x + w for x, w in zip(_mv(_mm(Zt, Fi), v), _mv(_t(L), r))]
        smooth[k] = [x + w for x, w in zip(at, _mv(Pt, r))]
    return RichResult(
        payload={
            "loglik": ll,
            "filtered": [k_[5] for k_ in keep],
            "smoothed": smooth,
            "predicted": [k_[0] for k_ in keep],
            "innovations": [k_[2][0] if p == 1 else k_[2] for k_ in keep],
            "F": [k_[6][0][0] if p == 1 else k_[6] for k_ in keep],
            "estimate": ll,
            "method": "Kalman filter and smoother (Durbin and Koopman 2012, ch. 4)",
        }
    )


def cheatsheet():
    return "stsmod: Kalman filter log-likelihood, filtered and smoothed states of y = Z a + e, a' = T a + R n"
