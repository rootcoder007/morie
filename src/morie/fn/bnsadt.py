"""Publication-bias correction and its adversarial bound (Andrews-Kasy).

A literature is not a random sample of the studies that were run. If
whether a result gets published depends on the result, then the
published estimates are draws from a TILTED distribution, and averaging
them -- however carefully -- averages the tilt along with the effect.

Andrews and Kasy's move is to write the tilt down as a parameter. Let
p(z) be the probability that a study with z-statistic z is published.
Then for a study with standard error sigma, the density of what we
actually observe is

    f(x | sigma) = p(x/sigma) * phi((x - mu)/s) / s / E[p]

with s^2 = tau^2 + sigma^2 for a latent effect Theta ~ N(mu, tau^2), and
E[p] the same numerator integrated over x. Because p is a step function
of z, E[p] is a finite sum of normal increments -- no quadrature, and no
approximation:

    E[p] = sum_k beta_k [ Phi((sigma c_{k+1} - mu)/s)
                          - Phi((sigma c_k - mu)/s) ]

That closed form is the reason the model is estimable from a meta-study
alone: the SAME mu and tau have to explain studies with very different
sigma, and only the selection can bend the relationship between them.

Two things then follow, and both are reported.

  The point estimate. Maximise the likelihood over (mu, tau, beta) and
  correct a single study's estimate to the MEDIAN-UNBIASED value: the
  theta for which the published X is equally likely to fall above and
  below the observed x,

      F(x | theta) = 1/2,   F(x | theta) = int_{-inf}^{x} p(u/sigma)
                                phi((u-theta)/sigma) du / (denominator)

  which is again a finite sum of normal increments and is solved by
  bisection. Median-unbiased rather than mean-unbiased because the
  truncated normal's mean does not exist in closed form and its median
  does; Andrews and Kasy make the same choice.

  The direction of the correction is toward zero at EVERY theta, not
  only for significant results. With symmetric selection and theta > 0
  the surviving upper tail carries more mass than the surviving lower
  tail, so the published median sits above theta wherever theta is; an
  insignificant published estimate is therefore shrunk too, not
  inflated. Only theta = 0 is a fixed point, and it is one exactly.

  The adversarial bound. The point estimate is conditional on ONE
  selection function. The honest object is what happens as p ranges over
  a whole family: the correction is monotone in the selection strength,
  so sweeping beta over the family gives an interval, and that interval
  -- not the point -- is what the data plus the family assumption
  supports. This is the "max over F" the module is named for.

Families, all selectable

  "none"             p == 1. No selection; the corrected estimate is the
                     observation itself, which is the check that the
                     machinery is not inventing a correction.
  "symmetric_step"   p = beta for |z| < 1.96, 1 above. One parameter.
  "symmetric_step2"  p = beta1 for |z| < 1.645, beta2 for 1.645 <= |z|
                     < 1.96, 1 above. Two parameters, and it is what
                     detects selection on the 10% level as well as the
                     5% one.
  "signed_step"      the paper's own minimum-wage specification: a
                     separate beta below -1.96, on (-1.96, 0), on
                     (0, 1.96), and 1 above 1.96. Four pieces, three
                     free. This is the family that can see selection on
                     the SIGN, which a symmetric family cannot.

Reference
  Andrews, I. and Kasy, M. (2019) "Identification of and Correction for
    Publication Bias." American Economic Review 109(8), 2766-2794.
    (Working-paper version arXiv:1711.10527.) The step-function
    specification, the meta-study likelihood, and the median-unbiased
    correction are all from there. The sweep over the family, and the
    reporting of the resulting interval, is this module's framing of
    their identification argument.
"""

import math

from . import _w3num as _w
from ._richresult import RichResult

__all__ = ["bound_adversarial", "bnsadt", "publication_probability",
           "selection_loglik", "median_unbiased", "fit_selection",
           "group_counts", "FAMILIES", "cheatsheet"]

# Each family is (signed cutoffs, one group index per interval). A group
# index of -1 means the interval's probability is fixed at 1, which is
# the normalisation: p is identified only up to scale, so the interval a
# significant result lands in carries the anchor.
FAMILIES = {
    "none": ((), (-1,)),
    "symmetric_step": ((-1.96, 1.96), (-1, 0, -1)),
    "symmetric_step2": ((-1.96, -1.645, 1.645, 1.96),
                        (-1, 1, 0, 1, -1)),
    "signed_step": ((-1.96, 0.0, 1.96), (0, 1, 2, -1)),
}


def _n_free(family):
    groups = FAMILIES[family][1]
    g = [k for k in groups if k >= 0]
    return 0 if not g else max(g) + 1


def _betas_for(family, params):
    """Per-interval publication probabilities from the free parameters."""
    groups = FAMILIES[family][1]
    return [1.0 if k < 0 else params[k] for k in groups]


def publication_probability(z, family="symmetric_step", params=()):
    """p(z), the probability a study with z-statistic z is published.

    Parameters
    ----------
    z : float
    family : str
        A key of FAMILIES.
    params : sequence
        The free probabilities, in group order.

    Returns
    -------
    float
    """
    cuts, _ = FAMILIES[family]
    betas = _betas_for(family, params)
    k = 0
    while k < len(cuts) and z >= cuts[k]:
        k += 1
    return betas[k]


def _expected_p(sigma, mu, tau, family, params):
    """E[p(X/sigma)] under the marginal law of an unselected study.

    A finite sum of normal increments because p is a step function --
    the reason no quadrature appears anywhere in this module.
    """
    cuts, _ = FAMILIES[family]
    betas = _betas_for(family, params)
    s = math.sqrt(tau * tau + sigma * sigma)
    edges = [-float("inf")] + [sigma * c for c in cuts] + [float("inf")]
    terms = []
    for k in range(len(betas)):
        lo, hi = edges[k], edges[k + 1]
        plo = 0.0 if lo == -float("inf") else _w.ncdf((lo - mu) / s)
        phi_ = 1.0 if hi == float("inf") else _w.ncdf((hi - mu) / s)
        terms.append(betas[k] * (phi_ - plo))
    return _w.csum(terms)


def selection_loglik(x, sigma, mu, tau, family="symmetric_step",
                     params=()):
    """Log-likelihood of the published estimates under the model."""
    n = len(x)
    terms = []
    for i in range(n):
        s = math.sqrt(tau * tau + sigma[i] * sigma[i])
        p = publication_probability(x[i] / sigma[i], family, params)
        if p <= 0.0:
            return float("-inf")
        d = _expected_p(sigma[i], mu, tau, family, params)
        if d <= 0.0:
            return float("-inf")
        z = (x[i] - mu) / s
        terms.append(math.log(p) - 0.5 * z * z
                     - math.log(s * math.sqrt(2.0 * math.pi))
                     - math.log(d))
    return _w.csum(terms)


def group_counts(x, sigma, family):
    """How many published studies fall in each free group's region.

    A group with NO observations is not identified from below: nothing
    in the data distinguishes "such studies are published one time in
    ten" from "one time in a billion", because none were seen either
    way. The likelihood is then monotone in that beta all the way to
    zero, and a maximiser sent after it walks off into the denormals --
    which is exactly what the two arms did differently before this
    existed.
    """
    groups = FAMILIES[family][1]
    cuts = FAMILIES[family][0]
    k = _n_free(family)
    counts = [0] * k
    for i in range(len(x)):
        z = x[i] / sigma[i]
        j = 0
        while j < len(cuts) and z >= cuts[j]:
            j += 1
        g = groups[j]
        if g >= 0:
            counts[g] += 1
    return counts


def fit_selection(x, sigma, family="symmetric_step", mu0=None, tau0=None,
                  beta0=0.5, iters=600):
    """Maximum likelihood over (mu, tau, beta) by Nelder-Mead.

    tau and every beta are optimised on the log scale, so the simplex
    cannot step to a negative variance or a negative probability and the
    run needs no penalty term to keep it inside the parameter space.
    Betas are then squashed by the logistic, which bounds them by 1 --
    a publication probability above the normalisation would be the
    likelihood saying the significant region is SUPPRESSED, which the
    family does not describe.

    Two guards, both of which are statements about identification and
    not numerical tape:

      A group with no observed studies is HELD at 1 and reported in
      `unidentified`, because its beta is bounded only from above.
      Optimising it anyway returns whatever the arithmetic drifts to.

      tau is floored at 1e-6 times the mean standard error. Below that
      the between-study spread is far under the within-study noise and
      the likelihood is flat in it; the floor is reported in
      `tau_at_floor` rather than hidden.
    """
    if family not in FAMILIES:
        raise ValueError("family must be one of %r" % (sorted(FAMILIES),))
    n = len(x)
    if mu0 is None:
        mu0 = _w.csum(x) / n
    if tau0 is None:
        v = _w.csum((xi - mu0) * (xi - mu0) for xi in x) / n
        w = _w.csum(si * si for si in sigma) / n
        tau0 = math.sqrt(v - w) if v > w else 0.5 * math.sqrt(v)
        if tau0 <= 0.0:
            tau0 = 0.1
    k = _n_free(family)
    counts = group_counts(x, sigma, family)
    active = [g for g in range(k) if counts[g] > 0]
    unident = [g for g in range(k) if counts[g] == 0]
    tau_floor = 1e-6 * (_w.csum(sigma) / n)

    def expand(par):
        ps = [1.0] * k
        for j, g in enumerate(active):
            ps[g] = 1.0 / (1.0 + math.exp(-par[2 + j]))
        tau = math.exp(par[1])
        return par[0], (tau if tau > tau_floor else tau_floor), ps

    start = [mu0, math.log(tau0)]
    start.extend([math.log(beta0 / (1.0 - beta0))] * len(active))

    def neg(par):
        mu, tau, ps = expand(par)
        ll = selection_loglik(x, sigma, mu, tau, family, ps)
        return -ll if ll == ll and ll != float("-inf") else 1e100

    r = _w.nelder_mead(neg, start, iters=iters)
    mu, tau, ps = expand(r["x"])
    return {"mu": mu, "tau": tau, "betas": ps, "loglik": -r["value"],
            "family": family, "n_free": len(active) + 2,
            "counts": counts, "unidentified": unident,
            "tau_at_floor": tau <= tau_floor,
            "tau_floor": tau_floor}


def _published_cdf(x, theta, sigma, family, params):
    """P(X <= x | Theta = theta), for a PUBLISHED study.

    Both the numerator and the denominator are finite sums of normal
    increments, so this is exact rather than quadrature.
    """
    cuts, _ = FAMILIES[family]
    betas = _betas_for(family, params)
    inf = float("inf")
    edges = [-inf] + [sigma * c for c in cuts] + [inf]
    num = []
    den = []
    for k in range(len(betas)):
        lo, hi = edges[k], edges[k + 1]
        clo = 0.0 if lo == -inf else _w.ncdf((lo - theta) / sigma)
        chi = 1.0 if hi == inf else _w.ncdf((hi - theta) / sigma)
        den.append(betas[k] * (chi - clo))
        if x > lo:
            top = hi if hi < x else x
            ctop = 1.0 if top == inf else _w.ncdf((top - theta) / sigma)
            num.append(betas[k] * (ctop - clo))
    d = _w.csum(den)
    if d <= 0.0:
        return float("nan")
    return _w.csum(num) / d


def median_unbiased(x, sigma, family="symmetric_step", params=(),
                    lo=None, hi=None):
    """The theta for which the published X has median x.

    Solved by bisection on a bracketing interval; the published CDF is
    decreasing in theta, so the root is unique.
    """
    if lo is None:
        lo = x - 20.0 * sigma
    if hi is None:
        hi = x + 20.0 * sigma
    # The published CDF decreases in theta, so f(lo) > 0 and f(hi) < 0;
    # bisect only needs a sign change, not a direction.
    return _w.bisect(
        lambda th: _published_cdf(x, th, sigma, family, params) - 0.5,
        lo, hi)


def bound_adversarial(y, D, family="symmetric_step", grid=None,
                      target=None, target_se=None, fit=True, iters=600):
    """Publication-bias correction with its adversarial bound.

    Parameters
    ----------
    y : sequence
        Published point estimates, one per study.
    D : sequence
        Their standard errors. Required and must be positive: the model
        is identified BY the variation in sigma across studies, so a
        constant or absent sigma is not a smaller version of this
        problem, it is a different one.
    family : str
        A key of FAMILIES.
    grid : sequence or None
        Publication probabilities to sweep for the adversarial bound.
        The default runs from 1 (no selection) down to 0.05, which spans
        "nothing was suppressed" to "insignificant work was almost never
        published".
    target : float or None
        The study to correct. Defaults to the largest |z| in the sample,
        which is the one selection distorts most and so the one the
        bound is most informative about.
    target_se : float or None
        Its standard error; taken from `D` when `target` is defaulted.
    fit : bool
        Estimate the selection parameters by maximum likelihood as well
        as sweeping the family.
    iters : int
        Nelder-Mead iterations.

    Returns
    -------
    RichResult
        The fit, the corrected estimate under it, and the adversarial
        interval over the family with the beta attaining each end.

    References
    ----------
    Andrews and Kasy (2019) AER 109(8), 2766-2794.
    """
    if family not in FAMILIES:
        raise ValueError("family must be one of %r" % (sorted(FAMILIES),))
    x = [float(v) for v in y]
    sigma = [float(v) for v in D]
    n = len(x)
    if len(sigma) != n:
        raise ValueError("y and D must have the same length")
    if any(s <= 0.0 for s in sigma):
        raise ValueError("standard errors must be positive")
    if n < 2:
        raise ValueError("a meta-study needs at least two studies")

    if target is None:
        j = 0
        for i in range(n):
            if abs(x[i] / sigma[i]) > abs(x[j] / sigma[j]):
                j = i
        target, target_se = x[j], sigma[j]
    elif target_se is None:
        raise ValueError("target_se is required when target is given")

    k = _n_free(family)
    res = {"family": family, "n": n, "target": float(target),
           "target_se": float(target_se),
           "target_z": float(target) / float(target_se),
           "method": "Andrews-Kasy publication-bias correction with an "
                     "adversarial bound over the selection family"}

    if fit:
        f = fit_selection(x, sigma, family, iters=iters)
        res["mu"] = f["mu"]
        res["tau"] = f["tau"]
        res["betas"] = f["betas"]
        res["group_counts"] = f["counts"]
        res["unidentified"] = f["unidentified"]
        res["tau_at_floor"] = f["tau_at_floor"]
        res["loglik"] = f["loglik"]
        res["estimate"] = median_unbiased(target, target_se, family,
                                          f["betas"])
        # The naive number the literature would report, for contrast.
        res["uncorrected"] = float(target)
        res["correction"] = res["estimate"] - float(target)
        # A likelihood-ratio statistic against no selection at all: the
        # nested model fixes every beta at 1.
        ll0 = selection_loglik(x, sigma, f["mu"], f["tau"], "none", ())
        res["loglik_no_selection"] = ll0
        res["lr_statistic"] = 2.0 * (f["loglik"] - ll0)
        res["lr_df"] = k - len(f["unidentified"])
        res["lr_p"] = (float("nan") if res["lr_df"] == 0 else
                       _w.gammq(res["lr_df"] / 2.0,
                                res["lr_statistic"] / 2.0))

    if grid is None:
        grid = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    sweep = []
    for b in grid:
        ps = [float(b)] * k
        sweep.append({"beta": float(b),
                      "estimate": median_unbiased(target, target_se,
                                                  family, ps)})
    ests = [s["estimate"] for s in sweep]
    lo_i = min(range(len(ests)), key=lambda i: (ests[i], i))
    hi_i = max(range(len(ests)), key=lambda i: (ests[i], -i))
    res["sweep"] = sweep
    res["bound_lower"] = ests[lo_i]
    res["bound_upper"] = ests[hi_i]
    res["bound_lower_beta"] = sweep[lo_i]["beta"]
    res["bound_upper_beta"] = sweep[hi_i]["beta"]
    res["bound_width"] = ests[hi_i] - ests[lo_i]
    res["se"] = float(target_se)
    return RichResult(payload=res)


bnsadt = bound_adversarial


def cheatsheet():
    return ("bnsadt: Andrews-Kasy publication-bias correction plus the "
            "adversarial bound over the selection family. families "
            + ", ".join(sorted(FAMILIES)))
