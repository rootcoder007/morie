# morie.fn -- function file (rootcoder007/morie)
"""Heteroskedastic IRT: per-legislator predictability (Lauderdale 2010)."""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._richresult import RichResult
from .sarreg import _brent

__all__ = ["heteroskedastic_irt"]

_SQ2 = math.sqrt(2.0)


def _Phi(z):
    return 0.5 * math.erfc(-z / _SQ2)


def _phi(z):
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _clip(p):
    return min(max(p, 1e-9), 1.0 - 1e-9)


def _probit_irls(Z, y, b0, prior_sd=None, max_iter=100):
    """Probit ML (or MAP with N(0, prior_sd^2) priors) without intercept by Fisher scoring."""
    b = list(b0)
    k = len(b)
    for _ in range(max_iter):
        eta = [ssum(z * c for z, c in zip(r, b)) for r in Z]
        A = [[0.0] * k for _ in range(k)]
        g = [0.0] * k
        for r, e, t in zip(Z, eta, y):
            m = _clip(_Phi(e))
            d = _phi(e)
            w = d * d / (m * (1.0 - m))
            s = d * (t - m) / (m * (1.0 - m))
            for a in range(k):
                g[a] += r[a] * s
                for c in range(k):
                    A[a][c] += w * r[a] * r[c]
        if prior_sd is not None:
            for a in range(k):
                A[a][a] += 1.0 / (prior_sd * prior_sd)
                g[a] -= b[a] / (prior_sd * prior_sd)
        det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
        if det == 0.0:
            break
        step = [(A[1][1] * g[0] - A[0][1] * g[1]) / det, (A[0][0] * g[1] - A[1][0] * g[0]) / det]
        b = [b[0] + step[0], b[1] + step[1]]
        if max(abs(v) for v in step) < 1e-12 * max(1.0, max(abs(v) for v in b)):
            break
    return b


def _nll_psi(t, idx, y, tau):
    e = math.exp(t)
    s = 0.0 if tau is None else 0.5 * t * t / (tau * tau)
    for a, v in zip(idx, y):
        p = _clip(_Phi(a / e))
        s -= v * math.log(p) + (1.0 - v) * math.log(1.0 - p)
    return s


def _score_psi(t, idx, y, tau):
    """d nll / d log psi (with the normal penalty on log psi when ``tau`` is set)."""
    e = math.exp(t)
    s = 0.0 if tau is None else t / (tau * tau)
    for a, v in zip(idx, y):
        z = a / e
        p = _Phi(z)
        if not 1e-9 < p < 1.0 - 1e-9:
            continue
        d = _phi(z)
        s -= (v * d / p - (1.0 - v) * d / (1.0 - p)) * (-z)
    return s


def heteroskedastic_irt(votes, ideal_points, item_params=None, max_iter=50, psi_prior_sd=1.0, item_prior_sd=5.0):
    r"""Estimate per-legislator noise scales given ideal points.

    Lauderdale's model divides each legislator's probit index by an
    individual scale:

    .. math:: P(y_{ij} = 1) = \Phi\!\Big(
              \frac{\beta_j x_i - \alpha_j}{\psi_i} \Big),

    so a large :math:`\psi_i` marks an *unpredictable* voter whose choices
    the spatial model explains poorly. With the ideal points fixed, this
    alternates (a) probit maximum likelihood of each item's ``(alpha_j,
    beta_j)`` given ``psi`` -- Fisher scoring of ``y_ij`` on ``(-1/psi_i,
    x_i/psi_i)`` without intercept, started at ``(0, 1)`` -- and (b) each
    legislator's ``log psi_i`` on ``[-3, 3]`` (Brent, then Newton on the
    score), and normalises ``psi`` to geometric mean 1 for identification,
    until ``max |psi change| < 1e-6``, then refits the items at the final scales. The unpenalised likelihood is
    degenerate: a voter predicted perfectly drives ``psi_i`` to 0 and the
    items then separate. Lauderdale estimates the model with priors; here the
    fit is the posterior mode under ``log psi_i ~ N(0, psi_prior_sd^2)`` and
    ``alpha_j, beta_j ~ N(0, item_prior_sd^2)`` (Albert 1992), the item
    step being penalised Fisher scoring; ``None`` drops a prior (plain ML). Probabilities are clipped to
    ``[1e-9, 1 - 1e-9]`` in the likelihood. Items or legislators with fewer
    than three observed votes or no variation are left at their start.

    Parameters
    ----------
    votes : array-like, shape (n, q)
        Binary votes (``None``/NaN = missing).
    ideal_points : array-like, shape (n,)
        Fixed ideal points.
    item_params : tuple (alpha, beta), optional
        Fixed item parameters.
    max_iter : int
        Alternation rounds.
    psi_prior_sd : float or None
        Prior standard deviation of ``log psi``.
    item_prior_sd : float or None
        Prior standard deviation of the item parameters.

    Returns
    -------
    RichResult
        ``psi`` (n), ``alpha`` (q), ``beta`` (q), ``loglik``, ``n``, ``q``,
        ``method``.

    References
    ----------
    Lauderdale, B. E. (2010). Unpredictable voters in ideal point estimation. *Political Analysis*,
    18(2), 151-171.

    Albert, J. H. (1992). Bayesian estimation of normal ogive item response curves using Gibbs sampling.
    *Journal of Educational Statistics*, 17(3), 251-269.

    Examples
    --------
    >>> V = [[1, 1, 0, 1, 0], [1, 0, 0, 1, 0], [0, 1, 1, 0, 1], [0, 0, 1, 1, 1], [1, 0, 1, 0, 1]]
    >>> r = heteroskedastic_irt(V, [-1.0, -0.5, 0.2, 0.9, 1.4], item_params=([0.1, -0.2, 0.3, 0.0, 0.5], [-1.0, -0.5, 1.2, -0.8, 1.0]))
    >>> [round(p, 6) for p in r["psi"]]
    [0.58473, 0.831694, 1.094527, 1.101132, 1.706141]
    """
    Vr = votes.tolist() if hasattr(votes, "tolist") else votes
    V = [[None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v) for v in row] for row in Vr]
    x = [float(v) for v in (ideal_points.tolist() if hasattr(ideal_points, "tolist") else ideal_points)]
    n = len(V)
    if n != len(x) or n == 0:
        raise ValueError("votes must be (n, q) with one ideal point per row.")
    q = len(V[0])
    for row in V:
        for v in row:
            if v is not None and v not in (0.0, 1.0):
                raise ValueError("votes must be binary 0/1 (NaN for missing).")
    if item_params is not None:
        alpha = [float(v) for v in item_params[0]]
        beta = [float(v) for v in item_params[1]]
        if len(alpha) != q or len(beta) != q:
            raise ValueError("item_params must be two length-q vectors.")
        fixed = True
    else:
        alpha, beta, fixed = [0.0] * q, [1.0] * q, False
    psi = [1.0] * n

    def fit_items():
        for j in range(q):
            rows = [i for i in range(n) if V[i][j] is not None]
            y = [V[i][j] for i in rows]
            if len(rows) < 3 or min(y) == max(y):
                alpha[j], beta[j] = 0.0, 0.0
                continue
            Z = [[-1.0 / psi[i], x[i] / psi[i]] for i in rows]
            alpha[j], beta[j] = _probit_irls(Z, y, [0.0, 1.0], item_prior_sd)

    for _ in range(int(max_iter)):
        if not fixed:
            fit_items()
        new = list(psi)
        for i in range(n):
            cols = [j for j in range(q) if V[i][j] is not None]
            y = [V[i][j] for j in cols]
            if len(cols) < 3 or min(y) == max(y):
                continue
            idx = [beta[j] * x[i] - alpha[j] for j in cols]
            t, _ = _brent(lambda s, idx=idx, y=y: _nll_psi(s, idx, y, psi_prior_sd), -3.0, 3.0)
            for _k in range(20):
                g = _score_psi(t, idx, y, psi_prior_sd)
                h = 1e-6
                dg = (_score_psi(t + h, idx, y, psi_prior_sd) - _score_psi(t - h, idx, y, psi_prior_sd)) / (2 * h)
                if not dg > 0.0:
                    break
                tn = t - g / dg
                if not -3.0 < tn < 3.0:
                    break
                step, t = abs(tn - t), tn
                if step < 1e-15:
                    break
            new[i] = math.exp(t)
        gm = math.exp(ssum(math.log(p) for p in new) / n)
        new = [p / gm for p in new]
        done = max(abs(a - b) for a, b in zip(new, psi)) < 1e-6
        psi = new
        if done:
            break
    if not fixed:
        fit_items()  # items at the returned scales
    ll = 0.0
    for i in range(n):
        for j in range(q):
            if V[i][j] is None:
                continue
            p = _clip(_Phi((beta[j] * x[i] - alpha[j]) / psi[i]))
            ll += V[i][j] * math.log(p) + (1.0 - V[i][j]) * math.log(1.0 - p)
    return RichResult(
        payload={
            "psi": psi,
            "alpha": alpha,
            "beta": beta,
            "loglik": ll,
            "n": n,
            "q": q,
            "method": "Heteroskedastic IRT scales given ideal points (Lauderdale 2010), posterior mode with normal priors",
        }
    )


def cheatsheet():
    return "hsirt: P = Phi((b x - a)/psi_i); alternate item probit ML and per-voter psi ML"
