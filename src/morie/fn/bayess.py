# morie.fn -- function file (rootcoder007/morie)
"""Effective sample size of an MCMC chain."""

from __future__ import annotations

from . import _array_core as np

from ._mcmc import ess_from_chains, split_rhat
from ._richresult import RichResult

__all__ = ["effective_sample_size_bayes"]


def effective_sample_size_bayes(chain):
    r"""Effective sample size: how many independent draws the chain is worth.

    .. math::
        n_{\text{eff}} = \frac{mn}{1 + 2\sum_{k\ge1}\rho_k},

    with the autocorrelation sum truncated by Geyer's initial positive
    sequence -- stopping at the first negative pair, which is what keeps the
    long-lag noise from dominating.

    ESS, not the raw draw count, is what determines Monte Carlo error:
    :math:`\mathrm{MCSE} \approx \mathrm{sd}/\sqrt{n_{\text{eff}}}`. A
    hundred thousand draws with ESS 50 carries the precision of 50 draws, and
    reporting the former is how a chain that has learned almost nothing gets
    presented as precise.

    ESS **above** the draw count is possible and not an error: antithetic
    behaviour gives negative autocorrelation, and NUTS produces it routinely.

    Parameters
    ----------
    chain : array-like
        Draws, ``(n,)`` for one chain or ``(m, n)`` for several.

    Returns
    -------
    RichResult
        ``ess``, ``n_draws``, ``efficiency``, ``mcse``, ``rhat``,
        ``autocorr_time``.

    References
    ----------
    Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Burkner, P.-C.
        (2021). Rank-normalization, folding, and localization: An improved
        R-hat for assessing convergence of MCMC. *Bayesian Analysis*,
        16(2), 667-718.

    Examples
    --------
    Independent draws have ESS close to the draw count.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> r = effective_sample_size_bayes(rng.normal(size=(4, 2000)))
    >>> bool(r["efficiency"] > 0.9)
    True

    A correlated chain is worth far less than its length. For AR(1) with
    rho = 0.9 the theoretical efficiency is (1-rho)/(1+rho) = 0.053.

    >>> x = np.zeros((4, 2500))
    >>> for j in range(4):
    ...     for i in range(1, 2500):
    ...         x[j, i] = 0.9 * x[j, i - 1] + rng.normal()
    >>> r2 = effective_sample_size_bayes(x[:, 500:])
    >>> bool(0.02 < r2["efficiency"] < 0.12)
    True

    Monte Carlo error follows ESS, not the number of draws.

    >>> bool(r2["mcse"] > r["mcse"])
    True

    >>> effective_sample_size_bayes([1.0, 2.0])
    Traceback (most recent call last):
        ...
    ValueError: need at least 4 draws per chain
    """
    C = np.atleast_2d(np.asarray(chain, dtype=float))
    m, n = C.shape
    if n < 4:
        raise ValueError("need at least 4 draws per chain")
    ess = ess_from_chains(C)
    total = m * n
    sd = float(np.std(C.ravel(), ddof=1))
    return RichResult(
        title="Effective sample size",
        summary_lines=[("draws", int(total)), ("ESS", float(ess)),
                       ("efficiency", float(ess / total))],
        warnings=(["ESS is below 100; the posterior summaries are dominated by "
                   "Monte Carlo error"] if ess < 100 else []),
        payload={
            "ess": float(ess), "n_draws": int(total),
            "efficiency": float(ess / total),
            "mcse": float(sd / np.sqrt(max(ess, 1e-12))),
            "rhat": split_rhat(C) if m > 1 else float("nan"),
            "autocorr_time": float(total / max(ess, 1e-12)),
            "n_chains": int(m), "method": "effective_sample_size_bayes",
        },
    )


def cheatsheet():
    return "bayess: MCSE follows ESS not draw count; ESS > n is fine (antithetic), ESS < 100 is not"
