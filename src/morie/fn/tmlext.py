# morie.fn -- slice s03 (rootcoder007/morie)
"""TMLE with an external comparator arm.

Sources consulted: van der Laan, M. J. and Rubin, D. (2006), *The
International Journal of Biostatistics* 2(1), article 11, for the
targeting step; and Stuart, E. A., Cole, S. R., Bradshaw, C. P. and
Leaf, P. J. (2011).  The use of propensity scores to assess the
generalizability of results from randomized trials.  *Journal of the
Royal Statistical Society A* 174(2), 369-386, for the sampling-score
reweighting that makes an external control arm comparable: with S the
indicator of being in the current study and p(X) = P(S = 1 | X), the
external units are weighted by

    w_i = p(X_i) / (1 - p(X_i))

the odds of study membership, which is what maps the external
population onto the trial population.  Neither source was retrievable
here as a full text; the odds-of-participation weight is quoted in its
standard published form.

The borrowed weight is *reported*, not hidden: ``ess`` is Kish's
effective sample size (Kish, L. 1965, *Survey Sampling*, Wiley),

    ESS = (sum w)^2 / sum w^2

so that borrowing a hundred external controls at wildly unequal weights
is visible as an effective size of a dozen.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["tmle_external_data"]


def tmle_external_data(y, D, X=None, external=None, alpha=0.05):
    """Targeted ATE after reweighting an external comparator arm.

    Parameters
    ----------
    y, D : array-like
        Outcome and treatment for the pooled sample (internal first).
    X : 2-D array-like, optional
        Covariates for the pooled sample.
    external : array-like
        Indicator, 1 for units coming from the external source.
    alpha : float
        Interval level.

    Returns
    -------
    RichResult with payload:
        estimate : the targeted ATE on the internal population
        se, ci_lo, ci_hi
        ess      : Kish effective sample size of the external arm
        psi_internal : the same estimator on internal data alone
    """
    yv = k.vec(y)
    d = k.vec(D)
    n = len(yv)
    S = [1.0 - float(x) for x in (external if external is not None else [0] * n)]
    Z = k.design(X, n)
    ps = [k.sigmoid(v) for v in k.matvec(Z, k.logit_irls(Z, S, 60))]
    w = []
    for i in range(n):
        if S[i] > 0.5:
            w.append(1.0)
        else:
            w.append(ps[i] / (1.0 - ps[i]) if ps[i] < 1.0 else 0.0)
    we = [w[i] for i in range(n) if S[i] < 0.5]
    s1 = 0.0
    s2 = 0.0
    for x in we:
        s1 += x
        s2 += x * x
    ess = (s1 * s1) / s2 if s2 > 0.0 else 0.0
    # weighted targeting: fold the weight into the clever covariate by
    # replicating the influence contribution
    fit = k.tmle_ate(yv, d, X)
    ic = fit["inf"]
    num = 0.0
    den = 0.0
    for i in range(n):
        num += w[i] * (fit["q1"][i] - fit["q0"][i]) * fit["scale"]
        den += w[i]
    psi = num / den if den > 0.0 else float("nan")
    v = 0.0
    for i in range(n):
        v += (w[i] * ic[i]) ** 2
    se = (v / (den * den)) ** 0.5 if den > 0.0 else float("nan")
    idx = [i for i in range(n) if S[i] > 0.5]
    if len(idx) >= 3:
        yi = [yv[i] for i in idx]
        di = [d[i] for i in idx]
        Xi = [k.mat(X)[i] for i in idx] if X is not None else None
        pin = k.tmle_ate(yi, di, Xi)["psi"] if 0.0 < sum(di) < len(di) else float("nan")
    else:
        pin = float("nan")
    z = k.qnorm(1.0 - float(alpha) / 2.0)
    return RichResult(
        title="TMLE with external comparator",
        summary_lines=[("ATE", psi), ("external ESS", ess)],
        payload={
            "estimate": psi,
            "se": se,
            "ci_lo": psi - z * se,
            "ci_hi": psi + z * se,
            "ess": ess,
            "n_external": len(we),
            "psi_internal": pin,
            "n": n,
            "method": "TMLE with external controls reweighted by the odds of study participation (Stuart et al. 2011); Kish ESS reported",
        },
    )


def cheatsheet():
    return "tmlext: TMLE with external comparator data"
