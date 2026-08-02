# morie.fn -- function file (rootcoder007/morie)
"""Imai-Keele-Tingley causal mediation under sequential ignorability."""

from . import _array_core as np

from ._richresult import RichResult
from .sensIM import _lsem_fit

__all__ = ["causal_mediation_imai"]


def causal_mediation_imai(x, m, y, c=None, n_boot=1000, seed=0, alpha=0.05):
    r"""ACME, ADE and total effect with nonparametric bootstrap intervals.

    Under sequential ignorability the average causal mediation effect
    and average direct effect are identified by the LSEM coefficients

    .. math:: \mathrm{ACME} = \gamma \beta_2, \qquad
              \mathrm{ADE} = \beta_3, \qquad
              \mathrm{TE} = \beta_1 = \mathrm{ACME} + \mathrm{ADE},

    where :math:`\beta_2` is the X -> M coefficient and
    :math:`\gamma, \beta_3` come from the outcome model. Intervals use
    the percentile bootstrap, which Imai et al. and MacKinnon et al.
    both prefer to the normal-theory Sobel interval because the
    product of two coefficients is skewed.

    Parameters
    ----------
    x, m, y : array-like, shape (n,)
        Treatment, mediator, outcome.
    c : array-like, optional
        Baseline covariates.
    n_boot : int, default 1000
        Bootstrap resamples.
    seed : int, default 0
        RNG seed.
    alpha : float, default 0.05
        Two-sided level for the percentile intervals.

    Returns
    -------
    RichResult
        keys: ``acme``, ``ade``, ``total``, ``prop_mediated``,
        ``acme_ci``, ``ade_ci``, ``total_ci``, ``n_boot``, ``n``,
        ``method``.

    References
    ----------
    Imai, K., Keele, L. & Tingley, D. (2010). A general approach to
    causal mediation analysis. *Psychological Methods*, 15(4),
    309-334.
    """
    f = _lsem_fit(x, m, y, c=c)
    acme = f["gamma"] * f["beta2"]
    ade = f["beta3"]
    total = acme + ade

    n = f["n"]
    nb = int(n_boot)
    if nb < 1:
        raise ValueError(f"n_boot must be at least 1, got {nb}.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}.")

    x = np.asarray(x, dtype=float).ravel()
    m = np.asarray(m, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    C = None if c is None else np.asarray(c, dtype=float).reshape(n, -1)

    rng = np.random.default_rng(seed)
    boot = np.empty((nb, 3))
    for i in range(nb):
        idx = rng.integers(0, n, n)
        bf = _lsem_fit(x[idx], m[idx], y[idx], c=None if C is None else C[idx])
        ac = bf["gamma"] * bf["beta2"]
        boot[i] = (ac, bf["beta3"], ac + bf["beta3"])

    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    ci = np.percentile(boot, [lo, hi], axis=0)

    return RichResult(
        payload={
            "acme": float(acme),
            "ade": float(ade),
            "total": float(total),
            "prop_mediated": float(acme / total) if total != 0 else float("nan"),
            "acme_ci": (float(ci[0, 0]), float(ci[1, 0])),
            "ade_ci": (float(ci[0, 1]), float(ci[1, 1])),
            "total_ci": (float(ci[0, 2]), float(ci[1, 2])),
            "n_boot": nb,
            "n": int(n),
            "method": "Causal mediation under sequential ignorability, percentile bootstrap",
        }
    )


def cheatsheet():
    return "causmedi: ACME = gamma*beta2, ADE = beta3, percentile-bootstrap CIs"
