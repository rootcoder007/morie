# morie.fn -- function file (rootcoder007/morie)
"""Tail effective sample size."""

from __future__ import annotations

import numpy as np

from ._mcmc import ess_from_chains
from ._richresult import RichResult

__all__ = ["effective_sample_size_tail"]


def effective_sample_size_tail(chains, prob=0.05):
    r"""Tail-ESS: effective sample size for the extreme quantiles.

    Computed as the minimum of the ESS of the indicator series
    :math:`\mathbb{1}\{x < q_{\alpha}\}` and
    :math:`\mathbb{1}\{x > q_{1-\alpha}\}`, so it measures how well the
    chain has explored each tail rather than the centre.

    This is the diagnostic that governs credible intervals, as bulk-ESS
    governs the mean. The two are **not** ordered in general and routinely
    differ by an order of magnitude in either direction: a threshold-crossing
    indicator can be far less autocorrelated than the series itself, so a
    sticky chain often has better tail-ESS than bulk-ESS, while a sampler that
    reaches the extremes only rarely has the reverse. Neither substitutes for
    the other, which is why both are reported and why an interval quoted on
    the strength of bulk-ESS alone is unsupported.

    Parameters
    ----------
    chains : array-like
        Draws ``(m, n)`` or ``(n,)``.
    prob : float
        Tail probability, in (0, 0.5).

    Returns
    -------
    RichResult
        ``ess_tail``, ``ess_lower``, ``ess_upper``, ``efficiency``,
        ``sufficient``.

    References
    ----------
    Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Burkner, P.-C.
        (2021). Rank-normalization, folding, and localization: An improved
        R-hat for assessing convergence of MCMC. *Bayesian Analysis*,
        16(2), 667-718.

    Examples
    --------
    Independent draws explore both tails well.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> r = effective_sample_size_tail(rng.normal(size=(4, 2000)))
    >>> bool(r["sufficient"])
    True

    The two diagnostics are not interchangeable and are not ordered. On a
    sampler stuck in separate modes, bulk-ESS collapses -- correctly, since
    the centre is not being explored -- while tail-ESS stays high, because
    crossing a tail threshold remains a comparatively frequent event.

    >>> y = np.zeros((4, 4000))
    >>> for j in range(4):
    ...     s = 1.0 if j % 2 else -1.0
    ...     for i in range(4000):
    ...         if rng.random() < 0.002:
    ...             s = -s
    ...         y[j, i] = s * 4 + rng.normal()
    >>> from morie.fn.essbk import effective_sample_size_bulk
    >>> bulk = effective_sample_size_bulk(y)["ess_bulk"]
    >>> tail = effective_sample_size_tail(y)["ess_tail"]
    >>> bool(bulk < 100 and tail > bulk)
    True

    Reporting only one of them would hide the pathology in whichever it is
    blind to.

    >>> bool(effective_sample_size_bulk(y)["sufficient"] is False)
    True

    Both tails are reported, since they can differ under a skewed posterior.

    >>> bool(r["ess_lower"] > 0 and r["ess_upper"] > 0)
    True
    """
    if not 0.0 < prob < 0.5:
        raise ValueError("prob must be in (0, 0.5)")
    C = np.atleast_2d(np.asarray(chains, dtype=float))
    m, n = C.shape
    if n < 4:
        raise ValueError("need at least 4 draws per chain")
    flat = C.ravel()
    q_lo, q_hi = np.quantile(flat, [prob, 1.0 - prob])
    # The indicator is already bounded and well behaved, so it is used
    # directly. Rank-normalising a 0/1 series maps every 0 to one value and
    # every 1 to another, which destroys the autocorrelation structure the
    # estimate depends on.
    lo = ess_from_chains((C < q_lo).astype(float))
    hi = ess_from_chains((C > q_hi).astype(float))
    ess = float(min(lo, hi))
    total = m * n
    return RichResult(
        title="Tail effective sample size",
        summary_lines=[("draws", int(total)), ("tail ESS", ess),
                       ("prob", float(prob))],
        warnings=([f"tail ESS below 100 per chain ({ess:.0f}); credible "
                   "intervals are unreliable even if the mean is fine"]
                  if ess < 100 * m else []),
        payload={
            "ess_tail": ess, "ess_lower": float(lo), "ess_upper": float(hi),
            "n_draws": int(total), "efficiency": float(ess / total),
            "sufficient": bool(ess >= 100 * m), "prob": float(prob),
            "n_chains": int(m), "method": "effective_sample_size_tail",
        },
    )


def cheatsheet():
    return "esstl: governs CREDIBLE INTERVALS; good bulk-ESS with bad tail-ESS means a trustworthy mean and a bad interval"
