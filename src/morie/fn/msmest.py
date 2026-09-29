# morie.fn -- function file (rootcoder007/morie)
"""Marginal structural model fit by inverse-probability-of-treatment weighting."""

import math

from ._qpcore import solve
from ._richresult import RichResult

__all__ = ["marginal_structural_model"]


def _rows(M):
    rows = M.tolist() if hasattr(M, "tolist") else list(M)
    return [[float(v) for v in (r if hasattr(r, "__len__") else [r])] for r in rows]


def _logit_probs(X, y, max_iter=100, tol=1e-9):
    """Newton-Raphson logistic regression of y on [1, X]; fitted probabilities."""
    D = [[1.0] + list(r) for r in X]
    n, k = len(D), len(D[0])
    beta = [0.0] * k

    def probs(b):
        out = []
        for r in D:
            eta = min(max(math.fsum(u * v for u, v in zip(r, b)), -35.0), 35.0)
            out.append(1.0 / (1.0 + math.exp(-eta)))
        return out

    for _ in range(max_iter):
        p = probs(beta)
        w = [max(q * (1 - q), 1e-10) for q in p]
        g = [math.fsum(D[i][a] * (y[i] - p[i]) for i in range(n)) for a in range(k)]
        H = [[math.fsum(w[i] * D[i][a] * D[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
        step = solve(H, g)
        beta = [u + v for u, v in zip(beta, step)]
        if max(abs(v) for v in step) < tol:
            break
    return probs(beta)


def marginal_structural_model(y, treatment_history, covariate_history):
    r"""Fit the MSM :math:`E[Y(\bar a)] = \beta_0 + \beta_1 \sum_t a_t` by IPTW.

    Computes stabilised inverse-probability-of-treatment weights

    .. math:: sw_i = \prod_t
              \frac{f(A_{it} \mid \bar A_{i,t-1})}
                   {f(A_{it} \mid \bar A_{i,t-1}, L_{it})},

    with both densities modelled by logistic regression (Newton-Raphson;
    the denominator probabilities clipped to [1e-6, 1 - 1e-6]), then fits the
    marginal structural model by weighted least squares of ``y`` on
    cumulative treatment. Under sequential exchangeability given the
    measured time-varying confounders :math:`L_t`, the weighted
    coefficient consistently estimates the causal per-period effect --
    which naive covariate adjustment cannot do when :math:`L_t` is
    itself affected by earlier treatment.

    Parameters
    ----------
    y : array-like, shape (n,)
        End-of-follow-up outcome.
    treatment_history : array-like of {0, 1}, shape (n, T) or (n,)
        Treatment at each interval.
    covariate_history : array-like, shape (n, T) or (n,)
        Time-varying confounder measured at the start of each interval.

    Returns
    -------
    RichResult
        keys: ``estimate`` (per-period causal effect beta_1),
        ``intercept``, ``weights`` (stabilised, n,), ``ess``, ``n``,
        ``n_periods``, ``method``.

    References
    ----------
    Robins, J. M., Hernan, M. A. & Brumback, B. (2000). Marginal
    structural models and causal inference in epidemiology.
    *Epidemiology*, 11(5), 550-560.

    Examples
    --------
    >>> import math
    >>> L = [[math.sin(i), math.cos(0.7 * i)] for i in range(40)]
    >>> A = [[1.0 if L[i][0] + 1.5 * math.sin(3.3 * i + 1) > 0 else 0.0,
    ...       1.0 if L[i][1] + 1.5 * math.cos(2.1 * i) > 0 else 0.0] for i in range(40)]
    >>> y = [1.0 + 0.5 * (A[i][0] + A[i][1]) + L[i][0] + 0.2 * math.sin(5 * i) for i in range(40)]
    >>> round(marginal_structural_model(y, A, L)["estimate"], 8)
    0.53574926
    """
    yv = [float(v) for v in (y.tolist() if hasattr(y, "tolist") else y)]
    A = _rows(treatment_history)
    L = _rows(covariate_history)
    n, T = len(A), len(A[0])
    if len(yv) != n or len(L) != n or len(L[0]) != T:
        raise ValueError(f"shapes disagree: y {len(yv)}, A ({n}, {T}), L ({len(L)}, {len(L[0])}).")
    if any(v not in (0.0, 1.0) for r in A for v in r):
        raise ValueError("treatment_history must be binary 0/1.")
    sw = [1.0] * n
    for t in range(T):
        a = [r[t] for r in A]
        if min(a) == max(a):
            continue  # no variation: both densities are the indicator, ratio 1
        if t > 0:
            p_num = _logit_probs([r[:t] for r in A], a)
        else:
            m = math.fsum(a) / n
            p_num = [m] * n
        p_den = [min(max(v, 1e-6), 1 - 1e-6) for v in _logit_probs([A[i][:t] + L[i][: t + 1] for i in range(n)], a)]
        for i in range(n):
            num = p_num[i] if a[i] == 1 else 1 - p_num[i]
            den = p_den[i] if a[i] == 1 else 1 - p_den[i]
            sw[i] *= num / den
    cum = [math.fsum(r) for r in A]
    sw0 = math.fsum(sw)
    s1 = math.fsum(w * c for w, c in zip(sw, cum))
    s2 = math.fsum(w * c * c for w, c in zip(sw, cum))
    ty = math.fsum(w * v for w, v in zip(sw, yv))
    tcy = math.fsum(w * c * v for w, c, v in zip(sw, cum, yv))
    det = sw0 * s2 - s1 * s1
    b1 = (sw0 * tcy - s1 * ty) / det
    b0 = (ty - b1 * s1) / sw0
    ess = sw0**2 / math.fsum(w * w for w in sw)
    return RichResult(
        payload={
            "estimate": b1,
            "intercept": b0,
            "weights": sw,
            "ess": ess,
            "n": n,
            "n_periods": T,
            "method": "Marginal structural model fit by IPTW (stabilised weights)",
        }
    )


def cheatsheet():
    return "msmest: MSM E[Y(abar)] = b0 + b1*sum(a_t) by stabilised IPTW WLS"
