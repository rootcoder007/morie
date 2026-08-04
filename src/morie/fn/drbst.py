# morie.fn -- slice s03 (rootcoder007/morie)
"""Bootstrap inference for the doubly robust DiD estimator.

Source consulted (FETCHED): Sant'Anna, P. H. C. and Zhao, J. (2020).
Doubly robust difference-in-differences estimators.  *Journal of
Econometrics* 219(1), 101-122 (arXiv:1812.01723).  Their equation (2.6)
is the panel estimand

    tau^(dr,p) = E[ ( w_1^p(D) - w_0^p(D, X; pi) )
                    ( dY - mu_(0,dY)^p(X) ) ]

with, equation (2.7),

    w_1^p(D)       = D / E[D]
    w_0^p(D, X; g) = [ g(X)(1-D) / (1-g(X)) ]
                     / E[ g(X)(1-D) / (1-g(X)) ]

and inference is by the *multiplier* bootstrap on the influence
function, which is what section 3.2 recommends over the empirical
bootstrap.

DETERMINISM.  Mammen's two-point multiplier is used, but the
"draw" is a van der Corput point rather than a pseudo-random one: the
multiplier still has mean 1, variance 1 and third moment 1, and both
arms produce the identical replicate sequence.  Nothing here resamples
indices at random.  The standard error is reported both as the
replicate standard deviation and, as the paper's companion software
does, as the interquartile range normalised by the corresponding normal
quantiles, (q75 - q25) / (z75 - z25), which is robust to the occasional
extreme replicate.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["dr_did_bootstrap"]


def dr_did_bootstrap(y, D, X=None, B=199, alpha=0.05, y0=None):
    """DR-DiD point estimate with multiplier-bootstrap inference.

    Parameters
    ----------
    y : array-like
        Either the outcome change dY = Y_1 - Y_0, or the period-1
        outcome when ``y0`` is supplied.
    D : array-like
        Treatment indicator.
    X : 2-D array-like, optional
        Covariates (an intercept is added).
    B : int
        Number of bootstrap replicates.
    alpha : float
        Two-sided level for the confidence interval.
    y0 : array-like, optional
        Period-0 outcome.

    Returns
    -------
    RichResult with payload:
        estimate : tau, the DR ATT
        se       : bootstrap standard error (IQR-normalised)
        se_sd    : replicate standard deviation
        ci_lo, ci_hi : symmetric normal interval
        boot     : the replicate values
    """
    dy = k.vec(y)
    if y0 is not None:
        y00 = k.vec(y0)
        dy = [dy[i] - y00[i] for i in range(len(dy))]
    fit = k.drdid_panel(dy, D, X)
    inf = fit["inf"]
    n = len(inf)
    boot = []
    for b in range(int(B)):
        s = 0.0
        for i in range(n):
            s += k.mammen(b * n + i) * inf[i]
        boot.append(fit["tau"] + s / n)
    q25 = k.quantile7(boot, 0.25)
    q75 = k.quantile7(boot, 0.75)
    z25 = k.qnorm(0.25)
    z75 = k.qnorm(0.75)
    se = (q75 - q25) / (z75 - z25) if boot else float("nan")
    se_sd = k.sd(boot, 1) if len(boot) > 1 else float("nan")
    z = k.qnorm(1.0 - float(alpha) / 2.0)
    return RichResult(
        title="DR-DiD with multiplier bootstrap",
        summary_lines=[("ATT", fit["tau"]), ("SE", se)],
        payload={
            "estimate": fit["tau"],
            "se": se,
            "se_sd": se_sd,
            "se_analytic": fit["se"],
            "ci_lo": fit["tau"] - z * se,
            "ci_hi": fit["tau"] + z * se,
            "boot": boot,
            "n": n,
            "B": int(B),
            "method": "DR-DiD (Sant'Anna and Zhao 2020, eq. 2.6) with a deterministic Mammen multiplier bootstrap",
        },
    )


def cheatsheet():
    return "drbst: Bootstrap inference for DR-DiD"


drdidbootstrap = dr_did_bootstrap
