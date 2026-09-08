# morie.fn -- function file (rootcoder007/morie)
r"""How much did the prior do, and did the data disagree with it?

Two questions get asked of a Bayesian analysis and answered by the same
comparison. The first is how *informative* the prior was: if the
posterior is the prior with a slightly different label, the data
contributed nothing and the conclusion is an assumption. The second is
whether there is *prior-data conflict*: if the data have pushed the
parameter somewhere the prior said was nearly impossible, the model is
in trouble whatever the posterior looks like -- Evans and Moshonov's
point is that this is a check on the prior that must be done before the
posterior is interpreted, not after.

The informativeness side is the Kullback-Leibler divergence of the
posterior from the prior,

.. math:: D_{\mathrm{KL}}(p(\theta\mid y)\,\|\,p(\theta))
          = \int p(\theta\mid y)\log\frac{p(\theta\mid y)}{p(\theta)}\,
            d\theta ,

computed two ways because they fail differently. The Gaussian route
matches the first two moments and is exact when both distributions are
normal, cheap, and blind to shape. The kernel route smooths both sets of
draws with a Gaussian kernel at Silverman's bandwidth and integrates the
divergence on a grid; it sees a bimodal posterior that the Gaussian
route reports as a wide unimodal one, at the cost of a bandwidth choice
that oversmooths sharp features. Both are always returned, and
``kl_divergence`` is the Gaussian one, whose failure mode is at least
predictable.

A nearest-neighbour estimator was the obvious third route and is
deliberately absent: it assumes the draws are independent, and the
deterministic low-discrepancy sequences this package uses in place of a
random number generator are far more evenly spaced than independent
draws, which biases every spacing-based estimate downward -- on normal
pairs with a known divergence of 0.5 it returns roughly -1. A number
that is wrong by a constant on the only inputs available is worse than
no number.

The conflict side is a tail probability: where the posterior mean falls
in the prior. A value in the far tail means the prior was placing almost
no mass where the data ended up.

Prior variance that survives into the posterior is quantified by the
shrinkage :math:`1-\operatorname{Var}(\theta\mid y)/\operatorname
{Var}(\theta)`, which is near 0 when the prior dominated and near 1 when
the data did.

References
----------
Evans, M. and Moshonov, H. (2006) "Checking for prior-data conflict",
*Bayesian Analysis* **1**(4), 893-914, doi:10.1214/06-BA129.

Kullback, S. and Leibler, R. A. (1951) "On information and sufficiency",
*Annals of Mathematical Statistics* **22**(1), 79-86,
doi:10.1214/aoms/1177729694.

Silverman, B. W. (1986) *Density Estimation for Statistics and Data
Analysis*, Chapman & Hall, Sec. 3.4 (the rule-of-thumb bandwidth used
for the kernel route), doi:10.1201/9781315140919.

Perez-Cruz, F. (2008) "Kullback-Leibler divergence estimation of
continuous distributions", *IEEE International Symposium on Information
Theory*, 1666-1670, doi:10.1109/ISIT.2008.4595271. The
nearest-neighbour alternative, and its independence assumption -- the
reason it is not used here.

Betancourt, M. (2017) "A conceptual introduction to Hamiltonian Monte
Carlo", arXiv:1701.02434. Prior-posterior shrinkage as a diagnostic of
how much the likelihood contributed.

Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A. and
Rubin, D. B. (2013) *Bayesian Data Analysis*, 3rd ed., CRC Press,
Sec. 6.3 (posterior predictive and prior predictive checks).

Nott, D. J., Wang, X., Evans, M. and Englert, B.-G. (2020) "Checking for
prior-data conflict using prior-to-posterior divergences", *Statistical
Science* **35**(2), 234-253, doi:10.1214/19-STS731.
"""

import math

from . import _s03core as k
from ._richresult import RichResult

__all__ = ["prior_informativeness_bias_diagnostic"]

_EPS = 1e-12


def _moments(v):
    n = len(v)
    m = sum(v) / n
    s2 = sum((x - m) ** 2 for x in v) / max(n - 1, 1)
    return m, s2


def _kl_gaussian(mq, sq2, mp, sp2):
    """KL(q || p) for two normals -- exact when both are normal."""
    sp2 = max(sp2, 1e-300)
    sq2 = max(sq2, 1e-300)
    return (0.5 * math.log(sp2 / sq2)
            + (sq2 + (mq - mp) ** 2) / (2.0 * sp2) - 0.5)


def _bandwidth(v):
    """Silverman's rule of thumb, robustified by the interquartile range."""
    n = len(v)
    s = sorted(v)
    _m, s2 = _moments(v)
    sd = math.sqrt(max(s2, 0.0))
    iqr = _quantile(s, 0.75) - _quantile(s, 0.25)
    a = min(sd, iqr / 1.34) if iqr > 0.0 else sd
    if a <= 0.0:
        a = max(sd, 1e-8)
    return 0.9 * a * n ** (-0.2)


def _kde(x, sample, h):
    c = 1.0 / (len(sample) * h * math.sqrt(2.0 * math.pi))
    s = 0.0
    for v in sample:
        z = (x - v) / h
        if abs(z) < 38.0:
            s += math.exp(-0.5 * z * z)
    return c * s


def _kl_kde(q, p, n_grid):
    """KL(q || p) from Gaussian kernel densities integrated on a grid."""
    n, m = len(q), len(p)
    if n < 2 or m < 2:
        return float("nan")
    hq = _bandwidth(q)
    hp = _bandwidth(p)
    lo = min(min(q) - 4.0 * hq, min(p) - 4.0 * hp)
    hi = max(max(q) + 4.0 * hq, max(p) + 4.0 * hp)
    if hi - lo <= _EPS:
        return 0.0
    dx = (hi - lo) / n_grid
    fq, fp = [], []
    for i in range(n_grid):
        x = lo + (i + 0.5) * dx
        fq.append(_kde(x, q, hq))
        fp.append(_kde(x, p, hp))
    # renormalise on the grid so truncation does not masquerade as
    # divergence -- the two densities must each integrate to one here
    zq = sum(fq) * dx
    zp = sum(fp) * dx
    if zq <= _EPS or zp <= _EPS:
        return float("nan")
    tot = 0.0
    for i in range(n_grid):
        a = fq[i] / zq
        b = fp[i] / zp
        if a > 1e-300:
            tot += a * math.log(a / max(b, 1e-300)) * dx
    return tot


def _ecdf(sorted_v, x):
    n = len(sorted_v)
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_v[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo / float(n)


def prior_informativeness_bias_diagnostic(samples, prior, n_grid=512):
    r"""Prior informativeness and prior-data conflict from draws.

    Parameters
    ----------
    samples : array-like
        Posterior draws of a scalar parameter.
    prior : array-like or mapping
        Prior draws of the same parameter, or a mapping with ``mean``
        and ``sd`` if the prior is normal and only its moments are at
        hand. With moments only the kernel route and the
        empirical tail probability are unavailable and report NaN
        rather than a number computed from a distribution that was never
        supplied.
    n_grid : int
        Grid size for the kernel divergence and for the Wasserstein
        distance between the two empirical distributions.

    Returns
    -------
    RichResult
        ``kl_divergence`` (Gaussian route) and ``kl_divergence_kde``,
        ``shrinkage``, ``bias_in_prior_sd``, the ``conflict_p_value``
        and a ``verdict``.
    """
    q = [float(v) for v in k.vec(samples)]
    nq = len(q)
    if nq < 2:
        raise ValueError("pibmd: at least two posterior draws are needed to "
                         "estimate a variance")

    moments_only = False
    if isinstance(prior, dict):
        if "mean" not in prior or "sd" not in prior:
            raise ValueError("pibmd: a mapping prior must give 'mean' and "
                             "'sd'")
        mp = float(prior["mean"])
        sdp = float(prior["sd"])
        if sdp <= 0.0:
            raise ValueError("pibmd: the prior standard deviation must be "
                             "positive")
        sp2 = sdp * sdp
        p = []
        moments_only = True
    else:
        p = [float(v) for v in k.vec(prior)]
        if len(p) < 2:
            raise ValueError("pibmd: at least two prior draws are needed -- "
                             "pass {'mean': ..., 'sd': ...} for a normal "
                             "prior known only by its moments")
        mp, sp2 = _moments(p)

    mq, sq2 = _moments(q)
    if sp2 <= _EPS:
        raise ValueError("pibmd: the prior has no spread, so every "
                         "divergence from it is infinite")

    kl_g = _kl_gaussian(mq, sq2, mp, sp2)
    kl_kde = float("nan") if moments_only else _kl_kde(q, p, int(n_grid))
    # the reverse divergence: KL is not symmetric, and the prior-to-
    # posterior direction is the one that blows up when the posterior has
    # mass where the prior has none
    kl_rev = _kl_gaussian(mp, sp2, mq, sq2)
    sym = 0.5 * (kl_g + kl_rev)

    shrink = 1.0 - sq2 / sp2
    bias = (mq - mp) / math.sqrt(sp2)

    if moments_only:
        z = bias
        pval = 2.0 * min(k.pnorm(z), 1.0 - k.pnorm(z))
        pval_emp = float("nan")
        wass = float("nan")
    else:
        ps = sorted(p)
        f = _ecdf(ps, mq)
        pval_emp = 2.0 * min(f, 1.0 - f)
        z = bias
        pval = 2.0 * min(k.pnorm(z), 1.0 - k.pnorm(z))
        qs = sorted(q)
        ng = int(n_grid)
        if ng < 2:
            raise ValueError("pibmd: n_grid must be at least 2")
        wass = 0.0
        for i in range(ng):
            u = (i + 0.5) / ng
            wass += abs(_quantile(qs, u) - _quantile(ps, u))
        wass /= ng

    conflict = (pval if moments_only else pval_emp)
    verdict = ("prior-data conflict: the posterior mean sits in the tail of "
               "the prior" if conflict < 0.05 else
               ("no evidence of prior-data conflict" if conflict == conflict
                else "conflict not assessable from moments alone"))
    if shrink < 0.05:
        informative = ("the prior dominated -- the data barely narrowed it")
    elif shrink > 0.95:
        informative = "the data dominated -- the prior is nearly irrelevant"
    else:
        informative = "prior and data both contributed"

    return RichResult(payload={
        "estimate": kl_g, "kl_divergence": kl_g,
        "kl_divergence_kde": kl_kde,
        "kl_divergence_reverse": kl_rev, "kl_symmetric": sym,
        "shrinkage": shrink, "bias_in_prior_sd": bias,
        "wasserstein_1": wass,
        "conflict_p_value": conflict,
        "conflict_p_value_gaussian": pval,
        "conflict_p_value_empirical": pval_emp,
        "posterior_mean": mq, "posterior_var": sq2,
        "posterior_sd": math.sqrt(max(sq2, 0.0)),
        "prior_mean": mp, "prior_var": sp2,
        "prior_sd": math.sqrt(sp2),
        "n_posterior": nq, "n_prior": len(p),
        "moments_only": moments_only,
        "verdict": verdict, "informativeness": informative,
        "method": "prior-data conflict and prior informativeness: Gaussian "
                  "and kernel-density KL(posterior || prior), shrinkage, "
                  "and the tail probability of the posterior mean under the "
                  "prior (Evans & Moshonov 2006; Silverman 1986)",
        "note": "kl_divergence is the Gaussian route, which is exact for "
                "normal pairs and blind to shape; kl_divergence_kde is "
                "shape-aware and will disagree "
                "when the posterior is not unimodal -- the disagreement is "
                "the signal, not an error",
    })


def _quantile(sorted_v, u):
    """Type-7 quantile of an already-sorted vector."""
    n = len(sorted_v)
    if n == 1:
        return sorted_v[0]
    h = (n - 1) * u
    lo = int(math.floor(h))
    hi = min(lo + 1, n - 1)
    return sorted_v[lo] + (h - lo) * (sorted_v[hi] - sorted_v[lo])


def cheatsheet():
    return ("pibmd: prior_informativeness_bias_diagnostic(samples, prior) -> "
            "KL(posterior||prior) two ways, shrinkage and a prior-data "
            "conflict p-value (Evans & Moshonov 2006, Bayesian Analysis "
            "1:893-914)")
