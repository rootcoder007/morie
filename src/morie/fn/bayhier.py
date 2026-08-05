# morie.fn -- wave 2 slice f_00 (rootcoder007/morie)
"""Hierarchical pooling: no pooling, complete pooling, partial pooling.

Sources: Lindley, D. V. and Smith, A. F. M. (1972), "Bayes estimates for
the linear model", Journal of the Royal Statistical Society Series B
34(1), 1-18, doi:10.1111/j.2517-6161.1972.tb00885.x (citation verified
against Crossref -- note the pages are 1-18); and Gelman, A. et al.,
Bayesian Data Analysis, 3rd edition, Chapter 5, which is the treatment
the module specification names.  BDA3 itself was not retrievable here,
so the estimator is written in the standard published form of the
exchangeable normal-normal hierarchy.

For groups g = 1..G with n_g observations, within-group variance
sigma^2 and between-group variance tau^2, the group mean ybar_g has
sampling variance sigma_g^2 = sigma^2 / n_g, and the posterior mean of
the group effect is the precision-weighted compromise

    lambda_g   = tau^2 / (tau^2 + sigma_g^2) = 1 / (1 + sigma_g^2/tau^2)
    theta_g    = lambda_g ybar_g + (1 - lambda_g) mu

with mu the precision-weighted grand mean.  lambda_g is the shrinkage
factor named in the specification.

The two limits are the whole point and are checked as anchors:

  tau^2 -> infinity gives lambda_g = 1 and theta_g = ybar_g, which is
    *no pooling* -- every group estimated on its own data;
  tau^2 -> 0 gives lambda_g = 0 and theta_g = mu for every group, which
    is *complete pooling* -- the groups are one.

Partial pooling is everything between, and the amount of shrinkage is
not uniform: a group with few observations has a large sigma_g^2 and is
pulled hard toward mu, while a large group barely moves.  Applying one
shrinkage factor to every group -- the common shortcut -- is wrong
whenever the design is unbalanced, and the unbalanced fixture is in the
tests for that reason.

tau^2 is estimated by the one-way method of moments,
tau^2 = max(0, (MS_between - MS_within) / n_tilde).  The truncation at
zero is real: a negative variance estimate means the data show less
between-group spread than sampling alone would produce, and the honest
answer there is complete pooling, not a negative variance.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["hierarchical_pooling"]


def hierarchical_pooling(y, group, sigma2=None, tau2=None):
    """Partial-pooled group estimates and their shrinkage factors.

    Parameters
    ----------
    y : array-like
        The observations.
    group : array-like
        Group label per observation; any values, compared for equality.
    sigma2 : float, optional
        Within-group variance.  Estimated by the pooled within-group
        mean square when omitted.
    tau2 : float, optional
        Between-group variance.  Estimated by the one-way method of
        moments when omitted; pass a value to force a pooling regime.

    Returns
    -------
    theta : the partially pooled group estimates
    lambda_g : the shrinkage factor per group
    theta_nopool : the raw group means
    theta_pool : the precision-weighted grand mean
    mu, sigma2, tau2 : the hyperparameters used
    """
    yy = core.vec(y)
    n = len(yy)
    if n == 0:
        raise ValueError("hierarchical_pooling: y is empty")
    gl = list(group)
    if len(gl) != n:
        raise ValueError("hierarchical_pooling: y and group have different lengths")
    labs = []
    for v in gl:
        if v not in labs:
            labs.append(v)
    labs = sorted(labs, key=lambda v: str(v))
    G = len(labs)
    if G < 2:
        raise ValueError("hierarchical_pooling: need at least two groups")
    idx = {}
    for i, lb in enumerate(labs):
        idx[str(lb)] = i
    ng = [0] * G
    sg = [0.0] * G
    for i in range(n):
        j = idx[str(gl[i])]
        ng[j] += 1
        sg[j] += yy[i]
    for j in range(G):
        if ng[j] == 0:
            raise ValueError("hierarchical_pooling: a group has no observations")
    ybar = [sg[j] / ng[j] for j in range(G)]
    # pooled within-group mean square, the one-way ANOVA residual MS
    ssw = 0.0
    for i in range(n):
        d = yy[i] - ybar[idx[str(gl[i])]]
        ssw += d * d
    if sigma2 is None:
        if n - G <= 0:
            raise ValueError("hierarchical_pooling: no residual degrees of freedom for sigma2")
        s2 = ssw / (n - G)
    else:
        s2 = float(sigma2)
        if s2 < 0.0:
            raise ValueError("hierarchical_pooling: sigma2 must be non-negative")
    grand = 0.0
    for v in yy:
        grand += v
    grand = grand / n
    if tau2 is None:
        ssb = 0.0
        for j in range(G):
            d = ybar[j] - grand
            ssb += ng[j] * d * d
        msb = ssb / (G - 1)
        # n_tilde is the usual unbalanced correction; it is n/G when balanced
        sq = 0.0
        for j in range(G):
            sq += ng[j] * ng[j]
        ntil = (n - sq / n) / (G - 1)
        t2 = (msb - s2) / ntil if ntil > 0.0 else 0.0
        if t2 < 0.0:
            t2 = 0.0
    else:
        t2 = float(tau2)
        if t2 < 0.0:
            raise ValueError("hierarchical_pooling: tau2 must be non-negative")
    lam = []
    for j in range(G):
        sgj = s2 / ng[j]
        lam.append(t2 / (t2 + sgj) if (t2 + sgj) > 0.0 else 0.0)
    # precision-weighted grand mean, the posterior mean of the hyper-mean
    num = 0.0
    den = 0.0
    for j in range(G):
        w = 1.0 / (t2 + s2 / ng[j]) if (t2 + s2 / ng[j]) > 0.0 else 0.0
        num += w * ybar[j]
        den += w
    mu = num / den if den > 0.0 else grand
    theta = [lam[j] * ybar[j] + (1.0 - lam[j]) * mu for j in range(G)]
    return RichResult(
        title="Hierarchical pooling",
        summary_lines=[("G", G), ("tau2", t2), ("sigma2", s2)],
        payload={
            "theta": theta,
            "estimate": theta[0],
            "lambda_g": lam,
            "theta_nopool": ybar,
            "theta_pool": mu,
            "mu": mu,
            "sigma2": s2,
            "tau2": t2,
            "n_g": ng,
            "grand_mean": grand,
            "G": G,
            "n": n,
            "method": "normal-normal partial pooling, lambda_g = tau2/(tau2 + sigma2/n_g); Lindley and Smith (1972)",
        },
    )


def cheatsheet():
    return "bayhier: Hierarchical pooling (no/complete/partial)"


# compact alias per ledger/NAMING.md
hierarchicalpooling = hierarchical_pooling
