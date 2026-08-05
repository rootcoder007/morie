# morie.fn -- wave 2 slice x_0_01 (rootcoder007/morie)
"""Bayesian bootstrap: Dirichlet reweighting of the observed sample.

Rubin, D. B. (1981), "The Bayesian Bootstrap", *The Annals of
Statistics* 9(1), 130-134, doi:10.1214/aos/1176345338 (verified against
Crossref).

Each replicate draws w ~ Dirichlet(1,...,1) over the n observed values
and evaluates the statistic on the reweighted sample, giving a posterior
sample of the functional rather than a sampling distribution.  Unlike
Efron's bootstrap the weights are continuous, so no observation is ever
dropped and the replicate distribution has no atoms.

The default statistic is the weighted mean, for which the posterior
variance is available in closed form: with w ~ Dirichlet(1,...,1),
Var(w_i) = (n-1)/(n^2 (n+1)) and Cov(w_i, w_j) = -1/(n^2 (n+1)), so

    Var(sum w_i x_i) = sum_i (x_i - xbar)^2 / (n (n + 1)),

which is the anchor for this module -- it does not run through the
resampling loop at all.  ``var_closed`` reports it whenever the default
statistic is in use.
"""

from __future__ import annotations

import math

from . import _s03core as core

from ._richresult import RichResult
from .btdir import dirichlet_rows

__all__ = ["boot_bayesian"]


def _wmean(x, w):
    s = 0.0
    for i in range(len(x)):
        s += w[i] * x[i]
    return s


def boot_bayesian(x, stat=None, B=200, seed=1):
    """Bayesian bootstrap replicates of a weighted statistic.

    Parameters
    ----------
    x : array-like
        The observed sample.
    stat : callable, optional
        ``stat(x, w)``, the statistic on the reweighted sample.  Defaults
        to the weighted mean.
    B : int
        Number of posterior draws.
    seed : int
        Seed for the shared deterministic stream.

    Returns
    -------
    RichResult
        ``theta_b`` (the B replicates), ``estimate`` (their mean),
        ``se`` (their standard deviation), ``lo``/``hi`` (2.5% and 97.5%
        type-7 quantiles), ``var_closed`` (the closed-form posterior
        variance of the weighted mean, NaN when a custom statistic is
        supplied), ``n``, ``B``.
    """
    xx = core.vec(x)
    n = len(xx)
    if n < 1:
        raise ValueError("boot_bayesian: need at least one observation")
    if int(B) < 2:
        raise ValueError("boot_bayesian: need at least two replicates")
    W = dirichlet_rows(n, B, seed)
    f = _wmean if stat is None else stat
    theta = [float(f(xx, w)) for w in W]
    m = core.mean(theta)
    if n > 1:
        xb = core.mean(xx)
        vc = sum((u - xb) ** 2 for u in xx) / (n * (n + 1.0))
    else:
        vc = 0.0
    return RichResult(
        title="Bayesian bootstrap (Rubin 1981)",
        summary_lines=[("n", n), ("B", int(B)), ("estimate", m)],
        payload={
            "theta_b": theta,
            "estimate": m,
            "se": core.sd(theta, 1) if len(theta) > 1 else float("nan"),
            "lo": core.quantile7(theta, 0.025),
            "hi": core.quantile7(theta, 0.975),
            "var_closed": vc if stat is None else float("nan"),
            "n": n,
            "B": int(B),
            "method": "Rubin (1981) Ann. Statist. 9(1):130-134",
        },
    )


def cheatsheet():
    return "btbayes: Dirichlet(1,..,1) reweighting; posterior var of the mean is sum(x-xbar)^2/(n(n+1))"


# compact alias per ledger/NAMING.md
bootbayesian = boot_bayesian
