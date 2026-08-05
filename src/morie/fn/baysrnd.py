# morie.fn -- wave 2 slice f_00 (rootcoder007/morie)
"""Random-effects shrinkage -- the same method as :mod:`morie.fn.bayhier`.

The specification here is "random intercepts u_g ~ N(0, tau^2)", citing
Lindley and Smith (1972).  That is the exchangeable normal-normal
hierarchy, and its posterior mean is exactly the partial-pooling
estimator ``bayhier`` computes: the random intercept is
u_g = theta_g - mu = lambda_g (ybar_g - mu) with
lambda_g = tau^2/(tau^2 + sigma^2/n_g).  "Random-effects shrinkage" and
"hierarchical partial pooling" are two names for one estimator.

There is therefore exactly one implementation.  This module calls
``bayhier`` and reports the random effects themselves, u_g, rather than
the pooled group means; writing the arithmetic a second time would agree
with the first at 1e-9 forever and be indistinguishable from correct
work while doubling the surface under a second name.

Recorded in ledger/wave2/DUPMAP.tsv as baysrnd -> bayhier.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401

from ._richresult import RichResult

from .bayhier import hierarchical_pooling as _pool

__all__ = ["shrinkage_random"]


def shrinkage_random(y, X=None, group=None, sigma2=None, tau2=None):
    """Random intercepts u_g and their shrinkage factors.

    Parameters
    ----------
    y : array-like
        The observations.
    X : array-like, optional
        Accepted for interface compatibility.  If ``group`` is omitted,
        X is taken to hold the grouping labels, which is how the stub
        signature ordered the arguments.
    group : array-like, optional
        Group label per observation.
    sigma2, tau2 : float, optional
        Variance components; estimated when omitted.

    Returns
    -------
    u_g : the random intercepts, theta_g - mu
    theta : the partially pooled group means
    lambda_g : the shrinkage factors
    """
    g = group if group is not None else X
    if g is None:
        raise ValueError("shrinkage_random: a grouping vector is required")
    r = _pool(y, g, sigma2=sigma2, tau2=tau2)
    mu = r["mu"]
    u = [t - mu for t in r["theta"]]
    return RichResult(
        title="Random-effects shrinkage",
        summary_lines=[("G", r["G"]), ("tau2", r["tau2"])],
        payload={
            "u_g": u,
            "estimate": u[0],
            "theta": r["theta"],
            "lambda_g": r["lambda_g"],
            "theta_nopool": r["theta_nopool"],
            "mu": mu,
            "sigma2": r["sigma2"],
            "tau2": r["tau2"],
            "n_g": r["n_g"],
            "G": r["G"],
            "n": r["n"],
            "method": "u_g = lambda_g (ybar_g - mu); shared implementation with morie.fn.bayhier",
        },
    )


def cheatsheet():
    return "baysrnd: Random-effects shrinkage (shares bayhier's implementation)"


# compact alias per ledger/NAMING.md
shrinkagerandom = shrinkage_random
