# morie.fn -- function file (rootcoder007/morie)
r"""SAIGE: a score test that survives extreme case-control imbalance.

A biobank PheWAS runs tens of millions of variants against thousands of
phenotypes, and most of those phenotypes are rare -- case:control of
1:100 or worse. Two things break there, and they break in opposite
directions.

**Relatedness inflates.** Population structure and cryptic relatedness
make individuals non-independent, so a test that assumes independence
finds signal that is family, not biology. Linear mixed models fix that
for quantitative traits, but a binary trait is not Gaussian and an LMM
applied to one has inflated type I error in its own right.

**Imbalance breaks the asymptotics.** A logistic mixed-model score test
-- GMMAT's approach -- handles the binary outcome properly, but it
still assumes the score statistic is asymptotically Gaussian. With
1000 cases and 400,000 controls that assumption fails: the score
distribution is skewed, and the Gaussian tail is far too thin. The test
then reports p-values that are much too small, which at genome-wide
scale means thousands of false positives.

**The saddlepoint approximation is the fix, and the reason it works is
worth stating.** The normal approximation keeps two moments. The
saddlepoint approximation is built from the entire cumulant generating
function :math:`K(t) = \log E[e^{tS}]`, so it keeps *all* of them. For
the score :math:`S = \sum_i G_i (Y_i - \hat\mu_i)` the CGF is available
in closed form because the :math:`Y_i` are independent Bernoulli given
the fitted means:

.. math:: K(t) = \sum_i \Big[
          \log\big(1 - \hat\mu_i + \hat\mu_i e^{G_i t}\big)
          - G_i t \hat\mu_i \Big].

Solve :math:`K'(\hat t) = s` for the observed :math:`s`, and Lugannani
and Rice give the tail probability from :math:`\hat t`,
:math:`w = \mathrm{sgn}(\hat t)\sqrt{2(\hat t s - K(\hat t))}` and
:math:`v = \hat t\sqrt{K''(\hat t)}`:

.. math:: P(S > s) \approx 1 - \Phi(w)
          + \phi(w)\left(\frac{1}{v} - \frac{1}{w}\right).

That expression is exact in the Gaussian case and stays accurate far
into the tail when the Gaussian one does not, which is precisely the
regime a PheWAS lives in.

**Where the skew comes from.** The score contribution of a variant is
:math:`G_i(Y_i - \hat\mu_i)`. With balanced classes the positive and
negative contributions offset symmetrically. With 1:100 imbalance
almost every :math:`\hat\mu_i` is near zero, so a case carrying the
minor allele contributes a large positive term while a control
contributes a small negative one -- a long right tail that two moments
cannot describe.

**Cost matters as much as calibration.** A test that is correct but
:math:`O(MN^2)` cannot run on 400,000 samples. The variance-ratio
trick used here is the paper's: estimate the ratio between the
variance of the full mixed-model score and the variance of a score
computed without the relatedness matrix, once, on a subset of variants;
then reuse it. The expensive part is paid a fixed number of times
rather than per variant.

References
----------
Zhou, W., Nielsen, J. B., Fritsche, L. G., Dey, R., Gabrielsen, M. E.,
Wolford, B. N. et al. (2018) "Efficiently controlling for case-control
imbalance and sample relatedness in large-scale genetic association
studies", *Nature Genetics*, doi:10.1038/s41588-018-0184-y. The
motivation (unbalanced case-control ratios in biobank PheWAS, the
failure of LMM and of the Gaussian score test), the use of the
saddlepoint approximation to calibrate the logistic mixed-model score
test, and the computational strategy for large N. Volume and pages are
not printed in the accepted-article file held locally.

Dey, R., Schmidt, E. M., Abecasis, G. R. & Lee, S. (2017) "A fast and
accurate algorithm to test for binary phenotypes and its application
to PheWAS", *The American Journal of Human Genetics* 101(1), 37-49,
doi:10.1016/j.ajhg.2017.05.014. The saddlepoint-approximation score
test for unrelated samples that SAIGE extends to mixed models.

Lugannani, R. & Rice, S. (1980) "Saddle point approximation for the
distribution of the sum of independent random variables", *Advances in
Applied Probability* 12(2), 475-490, doi:10.2307/1426607. The tail
formula implemented in :func:`saddlepoint_pvalue`.

Chen, H., Wang, C., Conomos, M. P., Stilp, A. M., Li, Z., Sofer, T. et
al. (2016) "Control for population structure and relatedness for
binary traits in genetic association studies via logistic mixed
models", *The American Journal of Human Genetics* 98(4), 653-666,
doi:10.1016/j.ajhg.2016.02.012. GMMAT, the Gaussian-approximation
predecessor whose calibration SAIGE repairs.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["score_statistic", "cgf", "saddlepoint_pvalue",
           "normal_pvalue", "saige_test", "variance_ratio"]

_EPS = 1e-12


def _fit_null(y, X, ridge=1e-8):
    """Logistic null model; returns fitted means."""
    D = k.design(X, len(y))
    beta = k.logit_irls(D, y, ridge=ridge)
    mu = [k.sigmoid(sum(D[i][j] * beta[j] for j in range(len(beta))))
          for i in range(len(y))]
    return mu, beta


def score_statistic(y, G, mu):
    r"""The score :math:`S = \sum_i G_i(Y_i - \hat\mu_i)` and its
    variance under the null.
    """
    yv = [float(v) for v in k.vec(y)]
    gv = [float(v) for v in k.vec(G)]
    mv = [float(v) for v in k.vec(mu)]
    n = len(yv)
    if not (len(gv) == len(mv) == n):
        raise ValueError("saigeg: y, G and mu must agree in length "
                         "(%d, %d, %d)" % (n, len(gv), len(mv)))
    if any(not 0.0 < v < 1.0 for v in mv):
        raise ValueError("saigeg: fitted means must lie strictly in "
                         "(0, 1)")
    s = sum(gv[i] * (yv[i] - mv[i]) for i in range(n))
    var = sum(gv[i] * gv[i] * mv[i] * (1.0 - mv[i]) for i in range(n))
    if var <= _EPS:
        raise ValueError("saigeg: the score has zero variance -- the "
                         "variant is monomorphic or every fitted mean "
                         "is degenerate")
    return {"score": s, "variance": var, "n": n}


def cgf(t, G, mu, order=0):
    r"""The cumulant generating function of the score, and its first
    two derivatives.

    :math:`K(t) = \sum_i [\log(1 - \mu_i + \mu_i e^{G_i t})
    - G_i t \mu_i]`. Using it whole is what separates the saddlepoint
    approximation from the Gaussian one, which keeps only
    :math:`K''(0)`.
    """
    gv = [float(v) for v in k.vec(G)]
    mv = [float(v) for v in k.vec(mu)]
    tot = 0.0
    for i in range(len(gv)):
        gt = gv[i] * float(t)
        gt = max(-500.0, min(500.0, gt))
        e = math.exp(gt)
        d = 1.0 - mv[i] + mv[i] * e
        if d <= _EPS:
            raise ValueError("saigeg: the CGF diverged at t = %r"
                             % (t,))
        if order == 0:
            tot += math.log(d) - gv[i] * float(t) * mv[i]
        elif order == 1:
            tot += gv[i] * (mv[i] * e / d - mv[i])
        elif order == 2:
            tot += gv[i] * gv[i] * mv[i] * e * (1.0 - mv[i]) / (d * d)
        else:
            raise ValueError("saigeg: order must be 0, 1 or 2")
    return tot


def _solve_saddle(s, G, mu, lo=-50.0, hi=50.0, tol=1e-11, iters=200):
    """Find t with K'(t) = s, by bisection on the monotone K'."""
    fl = cgf(lo, G, mu, 1) - s
    fh = cgf(hi, G, mu, 1) - s
    if fl > 0 or fh < 0:
        raise ValueError("saigeg: the observed score %g lies outside "
                         "the range K'(t) can reach -- no saddlepoint "
                         "exists" % s)
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        fm = cgf(mid, G, mu, 1) - s
        if abs(fm) < tol:
            return mid
        if fm < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def saddlepoint_pvalue(s, G, mu, two_sided=True):
    r"""Lugannani-Rice tail probability for the score.

    Falls back to the Gaussian value in the immediate neighbourhood of
    the mean, where :math:`w \to 0` makes :math:`1/v - 1/w`
    numerically unstable -- that is the one region where the two agree
    anyway.
    """
    sv = float(s)
    var0 = cgf(0.0, G, mu, 2)
    if var0 <= _EPS:
        raise ValueError("saigeg: the score has zero variance")
    if abs(sv) < 1e-6 * math.sqrt(var0):
        p = 2.0 * (1.0 - k.pnorm(abs(sv) / math.sqrt(var0)))
        return {"p_value": min(1.0, p), "method": "normal (at the "
                "mean, where the saddlepoint is unstable and the two "
                "agree)", "t_hat": 0.0}
    that = _solve_saddle(sv, G, mu)
    kt = cgf(that, G, mu, 0)
    k2 = cgf(that, G, mu, 2)
    if k2 <= _EPS:
        raise ValueError("saigeg: K''(t) is non-positive at the "
                         "saddlepoint")
    inner = 2.0 * (that * sv - kt)
    w = math.copysign(math.sqrt(max(inner, 0.0)), that)
    v = that * math.sqrt(k2)
    if abs(w) < 1e-9 or abs(v) < 1e-12:
        p1 = 1.0 - k.pnorm(abs(sv) / math.sqrt(var0))
    else:
        phi = math.exp(-0.5 * w * w) / math.sqrt(2.0 * math.pi)
        p1 = 1.0 - k.pnorm(w) + phi * (1.0 / v - 1.0 / w)
    p1 = min(max(p1, 0.0), 1.0)
    p = 2.0 * min(p1, 1.0 - p1) if two_sided else p1
    return {"p_value": min(1.0, max(p, 0.0)), "t_hat": that,
            "w": w, "v": v, "K": kt, "K2": k2,
            "method": "saddlepoint (Lugannani-Rice), all cumulants"}


def normal_pvalue(s, variance, two_sided=True):
    """The Gaussian score p-value -- two moments only."""
    if float(variance) <= 0.0:
        raise ValueError("saigeg: the variance must be positive")
    z = float(s) / math.sqrt(float(variance))
    p = 2.0 * (1.0 - k.pnorm(abs(z))) if two_sided \
        else 1.0 - k.pnorm(z)
    return {"p_value": min(1.0, max(p, 0.0)), "z": z,
            "method": "normal approximation, first two moments"}


def variance_ratio(scores_full, scores_naive):
    r"""The paper's variance-ratio shortcut.

    The ratio between the variance of the full mixed-model score and
    that of a score ignoring relatedness is estimated once, on a
    subset of variants, and reused. The expensive computation is then
    paid a fixed number of times rather than per variant.
    """
    a = [float(v) for v in k.vec(scores_full)]
    b = [float(v) for v in k.vec(scores_naive)]
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("saigeg: need at least 2 matched score pairs")
    va = k.variance(a)
    vb = k.variance(b)
    if vb <= _EPS:
        raise ValueError("saigeg: the naive scores have zero variance")
    return {"ratio": va / vb, "var_full": va, "var_naive": vb,
            "n_variants": len(a)}


def saige_test(y, G, X=None, mu=None, ratio=1.0, two_sided=True):
    r"""Score test for one variant, calibrated by the saddlepoint.

    ``mu`` may be supplied from a previously fitted null model -- the
    point of the design is that the null is fitted **once** for a
    phenotype and reused across every variant.
    """
    yv = [float(v) for v in k.vec(y)]
    for v in yv:
        if v not in (0.0, 1.0):
            raise ValueError("saigeg: the phenotype must be 0/1, got "
                             "%r" % (v,))
    n_case = int(sum(yv))
    if n_case == 0 or n_case == len(yv):
        raise ValueError("saigeg: the phenotype has only one class "
                         "(%d cases of %d)" % (n_case, len(yv)))
    if mu is None:
        mu, _ = _fit_null(yv, X if X is not None else [[] for _ in yv])
    st = score_statistic(yv, G, mu)
    var = st["variance"] * float(ratio)
    nrm = normal_pvalue(st["score"], var, two_sided=two_sided)
    spa = saddlepoint_pvalue(st["score"], G, mu, two_sided=two_sided)
    return RichResult(payload={
        "estimate": spa["p_value"], "p_value": spa["p_value"],
        "p_normal": nrm["p_value"], "score": st["score"],
        "variance": var, "z": nrm["z"],
        "case_control_ratio": n_case / float(len(yv) - n_case),
        "n_cases": n_case, "n_controls": len(yv) - n_case,
        "variance_ratio": float(ratio),
        "saddlepoint": spa,
        "method": "logistic mixed-model score test with saddlepoint "
                  "calibration; Zhou et al. (2018)",
        "why": "the Gaussian approximation keeps two moments and is "
               "anti-conservative under case-control imbalance; the "
               "saddlepoint keeps all of them",
    })


def cheatsheet():
    return ("saigeg: SAIGE. Score S = sum G_i (Y_i - mu_i) from a "
            "logistic mixed model. Under 1:100 case-control imbalance "
            "S is right-skewed and the GAUSSIAN tail is far too thin, "
            "so p-values come out much too small. The saddlepoint "
            "approximation uses the whole CGF -- all cumulants -- via "
            "Lugannani-Rice, and stays calibrated in the tail. The "
            "variance ratio is estimated once and reused so the cost "
            "is not O(MN^2).")


# compact alias per ledger/NAMING.md
saigegwas = saige_test

# public names resolved by fn/_lazy_map.json
saige_gwas = saige_test
