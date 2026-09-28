# morie.fn -- function file (rootcoder007/morie)
"""Linear mixed models for longitudinal data with serial correlation and measurement error
(Diggle 1988; Diggle, Heagerty, Liang and Zeger 2002): serial correlation functions, the
marginal covariance ``V_i = sigma_b^2 J + tau^2 H_i(phi) + nu^2 I`` and its maximum likelihood /
REML fit with the fixed effects profiled out by generalised least squares."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import ssum
from ._richresult import RichResult
from .neldmd import nelder_mead

__all__ = ["serial_correlation", "lmm_serial_covariance", "lmm_serial_loglik", "lmm_serial_fit"]


def _vec(x):
    return [float(v) for v in np.asarray(x, dtype=float).ravel().tolist()]


def _mat(X):
    a = np.asarray(X, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return [[float(v) for v in r] for r in a.tolist()]


def serial_correlation(u, phi, kind="exponential"):
    r"""Serial correlation ``g(|u|)`` of a stationary within-subject process.

    ``"exponential"``: ``exp(-phi |u|)`` (continuous-time AR(1));
    ``"gaussian"``: ``exp(-phi u^2)``; ``"ar1"``: ``phi^|u|`` for equally spaced
    times (the latent ``alpha_t = phi alpha_{t-1} + eta_t``).

    References
    ----------
    Diggle, P. J., Heagerty, P., Liang, K.-Y. and Zeger, S. L. (2002). *Analysis
    of Longitudinal Data*, 2nd edn. Oxford University Press, section 5.2.

    Examples
    --------
    >>> [round(v, 12) for v in serial_correlation([0.0, 1.0, 2.0], 0.5)]
    [1.0, 0.606530659713, 0.367879441171]
    """
    out = []
    for v in _vec(u):
        a = abs(v)
        if kind == "exponential":
            out.append(math.exp(-phi * a))
        elif kind == "gaussian":
            out.append(math.exp(-phi * a * a))
        elif kind == "ar1":
            out.append(phi**a)
        else:
            raise ValueError("kind must be 'exponential', 'gaussian' or 'ar1'")
    return out


def lmm_serial_covariance(times, sigma_b2, tau2, phi, nu2, kind="exponential"):
    r"""Marginal covariance of one subject: ``sigma_b^2 J + tau^2 H(phi) + nu^2 I``.

    Random intercept variance ``sigma_b^2``, serial process variance ``tau^2``
    with correlation ``H_jk = g(|t_j - t_k|)`` and measurement-error variance
    ``nu^2`` (Diggle 1988, model ``Y_i = X_i beta + Z_i b_i + W_i(t) + e_i``).

    Examples
    --------
    >>> [[round(v, 12) for v in r] for r in lmm_serial_covariance([0.0, 1.0], 1.0, 2.0, 0.5, 0.25)]
    [[3.25, 2.213061319425], [2.213061319425, 3.25]]
    """
    t = _vec(times)
    n = len(t)
    return [
        [sigma_b2 + tau2 * serial_correlation([t[j] - t[k]], phi, kind)[0] + (nu2 if j == k else 0.0) for k in range(n)]
        for j in range(n)
    ]


def _chol(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - ssum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if s <= 0:
                    raise ValueError("covariance not positive definite")
                L[i][i] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L


def _fwd(L, b):
    x = []
    for i in range(len(L)):
        x.append((b[i] - ssum(L[i][k] * x[k] for k in range(i))) / L[i][i])
    return x


def _groups(subject):
    pos, idx = {}, []
    for s in list(subject):
        if s not in pos:
            pos[s] = len(pos)
        idx.append(pos[s])
    return idx, len(pos)


def _profile(y, X, idx, g, t, lam_b, phi, c, kind, reml):
    # whiten each subject with the Cholesky factor of Vtilde = lam_b J + (1 - c) H + c I
    p = len(X[0])
    ld = 0.0
    Xw, yw = [], []
    for s in range(g):
        rows = [i for i in range(len(y)) if idx[i] == s]
        V = lmm_serial_covariance([t[i] for i in rows], lam_b, 1.0 - c, phi, c, kind)
        L = _chol(V)
        ld += 2.0 * ssum(math.log(L[i][i]) for i in range(len(L)))
        yw += _fwd(L, [y[i] for i in rows])
        cols = [_fwd(L, [X[i][j] for i in rows]) for j in range(p)]
        Xw += [[cols[j][r] for j in range(p)] for r in range(len(rows))]
    xtx = [[ssum(r[a] * r[b] for r in Xw) for b in range(p)] for a in range(p)]
    xty = [ssum(r[a] * v for r, v in zip(Xw, yw)) for a in range(p)]
    Lx = _chol(xtx)
    z = _fwd(Lx, xty)
    beta = [0.0] * p
    for i in range(p - 1, -1, -1):
        beta[i] = (z[i] - ssum(Lx[k][i] * beta[k] for k in range(i + 1, p))) / Lx[i][i]
    rss = ssum((v - ssum(r[a] * beta[a] for a in range(p))) ** 2 for r, v in zip(Xw, yw))
    n = len(y)
    m = n - p if reml else n
    s2 = rss / m
    ll = -0.5 * (m * math.log(2 * math.pi * s2) + ld + m)
    if reml:
        ll -= ssum(math.log(Lx[i][i]) for i in range(p))
    return ll, beta, s2


def lmm_serial_loglik(y, X, subject, times, sigma_b2, tau2, phi, nu2, kind="exponential", reml=False):
    r"""Profile log-likelihood of the serial-correlation LMM at given variance parameters.

    The covariance parameters enter through the ratios to their total scale
    ``sigma^2 = tau^2 + nu^2``; ``beta`` and ``sigma^2`` are replaced by their
    GLS/ML (or REML) estimates, as in ``nlme::lme``. ``X`` should include the
    intercept column.

    Examples
    --------
    >>> y = [1.0, 2.0, 1.5, 3.0, 2.5, 4.0]
    >>> X = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]] * 2
    >>> r = lmm_serial_loglik(y, X, [1, 1, 1, 2, 2, 2], [0.0, 1.0, 2.0] * 2, 0.5, 0.8, 0.7, 0.2)
    >>> round(r.loglik, 10)
    -7.2725520811
    """
    yv, Xm = _vec(y), _mat(X)
    idx, g = _groups(subject)
    s = tau2 + nu2
    ll, beta, s2 = _profile(yv, Xm, idx, g, _vec(times), sigma_b2 / s, phi, nu2 / s, kind, reml)
    return RichResult(payload={"loglik": ll, "beta": beta, "sigma2": s2})


def lmm_serial_fit(y, X, subject, times, kind="exponential", reml=False, start=None):
    r"""Fit the Diggle (1988) LMM ``Y_i = X_i beta + b_i 1 + W_i(t) + e_i`` by ML or REML.

    ``b_i ~ N(0, sigma_b^2)``, ``W_i`` a stationary Gaussian process with
    variance ``tau^2`` and correlation ``g(u; phi)``, ``e_i ~ N(0, nu^2 I)``.
    The three ratios (``sigma_b^2 / sigma^2``, ``phi``, nugget share ``nu^2 /
    sigma^2``) are optimised by Nelder-Mead over ``(r_b, log phi, r_c)`` with
    ratio ``r_b^2`` and share ``r_c^2 / (1 + r_c^2)`` (so the boundary 0 is reachable), with
    ``beta`` and ``sigma^2`` profiled out; equivalent to ``nlme::lme`` with a
    random intercept and ``corExp``/``corGaus`` with a nugget.

    References
    ----------
    Diggle, P. J. (1988). An approach to the analysis of repeated measurements.
    *Biometrics* 44, 959-971.

    Pinheiro, J. C. and Bates, D. M. (2000). *Mixed-Effects Models in S and
    S-PLUS*. Springer, section 5.3.

    Examples
    --------
    >>> y = [1.0, 2.2, 1.4, 3.1, 2.4, 4.3, 0.2, 1.1, 1.9, 2.6, 3.9, 3.2]
    >>> X = [[1.0, t] for t in (0.0, 1.0, 2.0)] * 4
    >>> r = lmm_serial_fit(y, X, [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], [0.0, 1.0, 2.0] * 4)
    >>> round(r.beta[1], 6)
    0.4875
    """
    yv, Xm, tv = _vec(y), _mat(X), _vec(times)
    idx, g = _groups(subject)
    x0 = [1.0, 0.0, 0.5] if start is None else [float(v) for v in start]

    def f(th):
        lam, phi, c = th[0] ** 2, math.exp(th[1]), th[2] ** 2 / (1.0 + th[2] ** 2)
        try:
            return -_profile(yv, Xm, idx, g, tv, lam, phi, c, kind, reml)[0]
        except (ValueError, ZeroDivisionError, OverflowError):
            return math.inf

    opt = nelder_mead(f, x0, step=0.5, xtol=1e-11, ftol=1e-14, max_iter=6000)
    th = opt["x"]
    lam, phi, c = th[0] ** 2, math.exp(th[1]), th[2] ** 2 / (1.0 + th[2] ** 2)
    ll, beta, s2 = _profile(yv, Xm, idx, g, tv, lam, phi, c, kind, reml)
    return RichResult(
        payload={
            "beta": beta,
            "sigma_b2": lam * s2,
            "tau2": (1.0 - c) * s2,
            "nu2": c * s2,
            "phi": phi,
            "loglik": ll,
            "converged": opt["converged"],
        }
    )


def cheatsheet() -> str:
    return "serial_correlation / lmm_serial_covariance / lmm_serial_loglik / lmm_serial_fit -> Diggle LMM."
