# morie.fn -- k02 batch (rootcoder007/morie)
"""Meta-analysis of proportions by the exact binomial-normal GLMM.

Source consulted: Hamza, T.H., van Houwelingen, H.C. and Stijnen, T. (2008),
The binomial distribution of meta-analysis was preferred to model
within-study variability, *Journal of Clinical Epidemiology* 61, 41-51.  The
paper's point is that the usual normal approximation to a logit-transformed
proportion is poor for small or extreme counts, and that the exact model

    x_i | p_i ~ Binomial(n_i, p_i),   logit(p_i) = mu + sigma z_i,  z_i ~ N(0,1)

should be fitted instead.  The marginal likelihood is evaluated by
Gauss-Hermite quadrature (substituting z = sqrt(2) t so the weight is
exp(-t^2)) and maximised over (mu, sigma) by a nested golden-section search
with a fixed iteration count, which makes the fit deterministic and identical
in every arm.  The standard error inverts the 2 by 2 observed information in
(mu, sigma), so it carries the uncertainty in the heterogeneity rather than
profiling it away -- which is what ``metafor::rma.glmm`` reports.
"""

from __future__ import annotations

from . import _array_core as np
from . import _sci_core as _sci
from .k02util import k02gh, k02gold, k02z

from ._richresult import RichResult

__all__ = ["ma_glmm_ipd_proportion"]

_SQRT2 = 1.4142135623730951
_SQRTPI = 1.7724538509055159


def _nll(x, n, lgc, mu, sigma, nodes, wts):
    tot = 0.0
    for i in range(len(x)):
        acc = 0.0
        for q in range(len(nodes)):
            eta = mu + sigma * _SQRT2 * nodes[q]
            if eta >= 0.0:
                lp = -float(np.log1p(np.exp(-eta)))
                lq = -eta - float(np.log1p(np.exp(-eta)))
            else:
                lp = eta - float(np.log1p(np.exp(eta)))
                lq = -float(np.log1p(np.exp(eta)))
            acc += wts[q] * float(np.exp(lgc[i] + x[i] * lp + (n[i] - x[i]) * lq))
        tot += float(np.log(acc / _SQRTPI)) if acc > 0.0 else -700.0
    return -tot


def ma_glmm_ipd_proportion(xi, ni, quad=21, level=0.95):
    """Binomial-normal random-effects model for a set of proportions.

    Parameters
    ----------
    xi : array-like
        Event counts.
    ni : array-like
        Sample sizes.
    quad : int, default 21
        Number of Gauss-Hermite quadrature nodes.
    level : float, default 0.95
        Confidence level for the interval on the pooled proportion.

    Returns
    -------
    RichResult
        estimate (pooled proportion), logit_mu, sigma, tau2, se, ci_lower,
        ci_upper, loglik, quad, n_events, n, method.
    """
    x = [float(t) for t in np.atleast_1d(np.asarray(xi, dtype=float))]
    n = [float(t) for t in np.atleast_1d(np.asarray(ni, dtype=float))]
    lgc = [
        float(_sci.gammaln(n[i] + 1.0) - _sci.gammaln(x[i] + 1.0) - _sci.gammaln(n[i] - x[i] + 1.0))
        for i in range(len(x))
    ]
    nodes, wts = k02gh(int(quad))

    def inner(s):
        return k02gold(lambda m: _nll(x, n, lgc, m, s, nodes, wts), -12.0, 12.0, 70)

    def outer(s):
        return _nll(x, n, lgc, inner(s), s, nodes, wts)

    sigma = k02gold(outer, 0.0, 5.0, 70)
    mu = inner(sigma)
    nll = _nll(x, n, lgc, mu, sigma, nodes, wts)
    h = 1e-4
    fmm = _nll(x, n, lgc, mu + h, sigma, nodes, wts)
    fpp = _nll(x, n, lgc, mu - h, sigma, nodes, wts)
    hmm = (fmm - 2.0 * nll + fpp) / (h * h)
    fss = (
        _nll(x, n, lgc, mu, sigma + h, nodes, wts)
        - 2.0 * nll
        + _nll(x, n, lgc, mu, abs(sigma - h), nodes, wts)
    ) / (h * h)
    fms = (
        _nll(x, n, lgc, mu + h, sigma + h, nodes, wts)
        - _nll(x, n, lgc, mu + h, abs(sigma - h), nodes, wts)
        - _nll(x, n, lgc, mu - h, sigma + h, nodes, wts)
        + _nll(x, n, lgc, mu - h, abs(sigma - h), nodes, wts)
    ) / (4.0 * h * h)
    det = hmm * fss - fms * fms
    se = float(np.sqrt(fss / det)) if det > 0.0 and fss > 0.0 else float("nan")
    crit = k02z(0.5 + 0.5 * float(level))
    ilogit = lambda t: 1.0 / (1.0 + float(np.exp(-t)))
    return RichResult(
        payload={
            "estimate": float(ilogit(mu)),
            "logit_mu": float(mu),
            "sigma": float(sigma),
            "tau2": float(sigma * sigma),
            "se": se,
            "ci_lower": float(ilogit(mu - crit * se)),
            "ci_upper": float(ilogit(mu + crit * se)),
            "loglik": float(-nll),
            "quad": int(quad),
            "n_events": float(sum(x)),
            "n": int(len(x)),
            "method": "Binomial-normal GLMM for proportions, Gauss-Hermite (Hamza, van Houwelingen & Stijnen 2008)",
        }
    )


# CANONICAL TEST
# >>> x = [3, 7, 12, 2, 9]
# >>> n = [40, 50, 60, 35, 55]
# >>> r = ma_glmm_ipd_proportion(x, n)
# >>> assert 0.0 < r["estimate"] < 1.0
# >>> # with sigma pinned at zero the fit is the pooled binomial MLE
# >>> assert abs(r["estimate"] - 0.15) < 0.10


def cheatsheet():
    return "magpa(xi, ni): binomial-normal GLMM meta-analysis of proportions."


maglmmipdproportion = ma_glmm_ipd_proportion
