# morie.fn -- function file (rootcoder007/morie)
"""Heteroskedastic IRT: per-legislator predictability (Lauderdale 2010)."""

import numpy as np
from scipy import optimize, stats

from ._richresult import RichResult

__all__ = ["heteroskedastic_irt"]


def heteroskedastic_irt(votes, ideal_points, item_params=None, max_iter=50):
    r"""Estimate per-legislator noise scales given ideal points.

    Lauderdale's model divides each legislator's probit index by an
    individual scale:

    .. math:: P(y_{ij} = 1) = \Phi\!\Big(
              \frac{\beta_j x_i - \alpha_j}{\psi_i} \Big),

    so a large :math:`\psi_i` marks an *unpredictable* voter whose
    choices the spatial model explains poorly -- and whose votes a
    homoskedastic estimator over-weights. This implementation
    alternates (a) ML item fits given (x, psi) and (b) per-legislator
    ML of :math:`\psi_i` given the items, with
    :math:`\prod_i \psi_i = 1` for identification. Ideal points are
    taken as given (estimate them with :mod:`morie.fn.mcmpp` first);
    the full joint sampler is out of scope and the docstring says so.

    Parameters
    ----------
    votes : array-like, shape (n, q)
        Binary votes (NaN = missing).
    ideal_points : array-like, shape (n,)
        Fixed ideal points.
    item_params : tuple (alpha, beta), optional
        Fixed item parameters; estimated by probit ML when omitted.
    max_iter : int, default 50
        Alternation rounds.

    Returns
    -------
    RichResult
        keys: ``psi`` (n, unpredictability scales, geometric mean 1),
        ``alpha`` (q,), ``beta`` (q,), ``loglik``, ``n``, ``q``,
        ``method``.

    References
    ----------
    Lauderdale, B. E. (2010). Unpredictable voters in ideal point
    estimation. *Political Analysis*, 18(2), 151-171. (NOT AJPS --
    the placeholder's venue was wrong.)
    """
    V = np.asarray(votes, dtype=float)
    x = np.asarray(ideal_points, dtype=float).ravel()
    if V.ndim != 2 or V.shape[0] != x.size:
        raise ValueError("votes must be (n, q) with one ideal point per row.")
    n, q = V.shape
    obs = ~np.isnan(V)
    ok = V[obs]
    if not np.all(np.isin(ok, (0.0, 1.0))):
        raise ValueError("votes must be binary 0/1 (NaN for missing).")

    psi = np.ones(n)

    def fit_item(j, psi):
        m = obs[:, j]
        y = V[m, j]
        if y.min() == y.max() or m.sum() < 3:
            return 0.0, 0.0

        def nll(p):
            al, be = p
            z = (be * x[m] - al) / psi[m]
            pr = np.clip(stats.norm.cdf(z), 1e-9, 1 - 1e-9)
            return -np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr))

        res = optimize.minimize(nll, np.array([0.0, 1.0]), method="Nelder-Mead")
        return float(res.x[0]), float(res.x[1])

    if item_params is not None:
        alpha, beta = (np.asarray(v, dtype=float).ravel() for v in item_params)
        if alpha.size != q or beta.size != q:
            raise ValueError("item_params must be two length-q vectors.")
        fixed_items = True
    else:
        alpha = np.zeros(q)
        beta = np.ones(q)
        fixed_items = False

    for _ in range(int(max_iter)):
        if not fixed_items:
            for j in range(q):
                alpha[j], beta[j] = fit_item(j, psi)
        new_psi = psi.copy()
        for i in range(n):
            m = obs[i]
            y = V[i, m]
            if m.sum() < 3 or y.min() == y.max():
                continue
            idx = beta[m] * x[i] - alpha[m]

            def nll(logp):
                z = idx / np.exp(logp)
                pr = np.clip(stats.norm.cdf(z), 1e-9, 1 - 1e-9)
                return -np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr))

            res = optimize.minimize_scalar(nll, bounds=(-3, 3), method="bounded")
            new_psi[i] = float(np.exp(res.x))
        # identify: geometric mean of psi = 1
        new_psi = new_psi / np.exp(np.mean(np.log(new_psi)))
        if np.max(np.abs(new_psi - psi)) < 1e-6:
            psi = new_psi
            break
        psi = new_psi

    z = (x[:, None] * beta[None, :] - alpha[None, :]) / psi[:, None]
    pr = np.clip(stats.norm.cdf(z), 1e-9, 1 - 1e-9)
    ll = float(np.nansum(np.where(obs, V * np.log(pr) + (1 - V) * np.log(1 - pr), 0.0)))

    return RichResult(
        payload={
            "psi": psi,
            "alpha": alpha,
            "beta": beta,
            "loglik": ll,
            "n": int(n),
            "q": int(q),
            "method": "Heteroskedastic IRT scales given ideal points (Lauderdale 2010)",
        }
    )


def cheatsheet():
    return "hsirt: P = Phi((b x - a)/psi_i); alternate item ML and per-voter psi ML"
