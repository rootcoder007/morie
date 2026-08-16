"""Nonparametric Bayes quantile function -- the posterior of F^-1(q).

A quantile estimate that comes with a standard error borrowed from
asymptotics is answering a different question from the one usually
asked. What a Bayesian nonparametric model gives instead is the POSTERIOR
of the quantile: F is a random distribution, F^-1(q) is a functional of
it, and every posterior draw of F carries a draw of the quantile with
it. The spread of those draws is the uncertainty, with no appeal to a
limit and no assumption that the sampling distribution is symmetric --
which for an extreme quantile it is not.

Three routes, because there are three genuinely different objects here
and calling any one of them "the" Bayesian quantile would be a choice
hidden inside an implementation:

  "mixture" (default)
      For each retained sweep of the Dirichlet-process mixture, invert
      that sweep's mixture CDF at q. The result is a sample from the
      posterior of F^-1(q). Summarise it however you like; the module
      reports the mean, the standard deviation and an equal-tailed
      credible interval.

  "predictive"
      Average the CDF over sweeps FIRST and invert once. That is the
      quantile of the posterior predictive distribution, and it is not
      the posterior mean of the quantile -- inversion does not commute
      with averaging. It is the right answer to "what value will the
      next observation fall below with probability q", which is a
      prediction rather than an inference about F.

  "bayesian_bootstrap"
      Rubin's Bayesian bootstrap: put a Dirichlet(1, ..., 1) posterior
      on the weights of the observed points and read off the weighted
      empirical quantile. No mixture, no smoothing, no prior on the
      shape of F. It is the limiting case of the DP posterior as the
      concentration goes to zero, so it is the honest baseline: if the
      mixture route says something very different, that difference is
      the smoothing, and it should be visible rather than assumed away.

The mixture routes work from the retained component states of the slice
sampler in `slbpdg` -- the SAME sampler, called with keep_draws, not a
second one written here. Two samplers for one model is two places for a
bug to live.

One honest limitation is reported rather than hidden. The slice sampler
carries finitely many components per sweep and leaves an unbroken
remainder of the stick; the mixture CDF therefore integrates to
1 - rest, not to 1. The routes normalise by the carried mass and report
`min_mass_carried`, so a run where the remainder was not negligible
announces itself instead of quietly shifting the tails.

References
  Kottas, A. and Krnjajic, M. (2009) "Bayesian semiparametric modelling
    in quantile regression." Scandinavian Journal of Statistics 36(2),
    297-319. The Dirichlet-process mixture model for the error
    distribution whose quantile functional is the object here.
  Rubin, D.B. (1981) "The Bayesian bootstrap." Annals of Statistics
    9(1), 130-134.
  Walker, S.G. (2007); Kalli, Griffin and Walker (2011) -- the sampler,
    through morie.fn.slbpdg.
"""

import math

from . import _array_core as _core
from . import _w3num as _w
from .slbpdg import slice_break_dp
from ._richresult import RichResult

__all__ = ["bnp_percent_quantile", "bnppct", "mixture_cdf",
           "mixture_quantile", "expand_bracket", "ROUTES", "cheatsheet"]

ROUTES = ("mixture", "predictive", "bayesian_bootstrap")


def mixture_cdf(x, w, mu, s2):
    """sum_k w_k Phi((x - mu_k)/sqrt(s2_k)), unnormalised."""
    return _w.csum(w[k] * _w.ncdf((x - mu[k]) / math.sqrt(s2[k]))
                   for k in range(len(w)))


def expand_bracket(f, lo, hi, iters=60):
    """Widen [lo, hi] by doubling until f changes sign across it.

    A fixed doubling schedule, not a search: both arms then visit
    exactly the same endpoints and the bisection that follows starts
    from the same interval.
    """
    flo, fhi = f(lo), f(hi)
    k = 0
    while (flo > 0.0) == (fhi > 0.0) and k < int(iters):
        mid = 0.5 * (lo + hi)
        half = hi - mid
        lo = mid - 2.0 * half
        hi = mid + 2.0 * half
        flo, fhi = f(lo), f(hi)
        k += 1
    return lo, hi


def mixture_quantile(q, w, mu, s2, lo=None, hi=None):
    """Invert the mixture CDF at q, normalising by the carried mass.

    Bisection rather than Newton: the CDF is monotone but its derivative
    is a mixture of narrow normals, and a Newton step off a flat stretch
    between two well-separated components lands anywhere.

    The bracket is taken from the COMPONENTS, not from the data. A
    slice sampler instantiates components from the prior to cover the
    slice, and an inverse-gamma prior draw can be enormous; such a
    component leaves the CDF far short of one anywhere near the data,
    so a data-width bracket fails to contain the root. That is not a
    numerical nuisance -- it is the model saying this draw of F has a
    very heavy tail -- so the bracket follows the components and is
    then widened until it genuinely brackets.
    """
    mass = _w.csum(w)
    if mass <= 0.0:
        return float("nan")

    def f(x):
        return mixture_cdf(x, w, mu, s2) / mass - q

    if lo is None or hi is None:
        sds = [math.sqrt(v) for v in s2]
        blo = min(mu[k] - 40.0 * sds[k] for k in range(len(mu)))
        bhi = max(mu[k] + 40.0 * sds[k] for k in range(len(mu)))
        lo = blo if lo is None else min(lo, blo)
        hi = bhi if hi is None else max(hi, bhi)
    lo, hi = expand_bracket(f, lo, hi)
    return _w.bisect(f, lo, hi)


def _weighted_quantile(ys_sorted, weights, q):
    """Weighted empirical quantile: the smallest point whose cumulative
    weight reaches q. The inverse of the weighted ECDF, which is a step
    function, so no interpolation is invented between the steps."""
    acc = 0.0
    for i in range(len(ys_sorted)):
        acc += weights[i]
        if acc >= q:
            return ys_sorted[i]
    return ys_sorted[-1]


def bnp_percent_quantile(y, quantile=0.5, route="mixture", alpha=1.0,
                         n_iter=500, burn=None, thin=1, seed=1,
                         cred=0.9, sampler_route="walker", kappa=0.5,
                         m0=None, kappa0=0.01, a0=2.0, b0=None,
                         n_bootstrap=500, alpha_update=None):
    """Posterior of the quantile function of a nonparametric F.

    Parameters
    ----------
    y : sequence
        The data.
    quantile : float or sequence
        One or more probabilities in (0, 1).
    route : str
        "mixture", "predictive" or "bayesian_bootstrap".
    alpha : float
        Dirichlet-process concentration for the mixture routes.
    n_iter, burn, thin : int
        Sampler length, burn-in and thinning.
    seed : int
        Seed for the generator shared with the R arm.
    cred : float
        Credible level for the reported interval.
    sampler_route : str
        Which slice sampler `slbpdg` should use.
    kappa, m0, kappa0, a0, b0 :
        Passed through to the sampler.
    n_bootstrap : int
        Replicates for the Bayesian-bootstrap route.
    alpha_update : str or None
        Passed through to the sampler.

    Returns
    -------
    RichResult
        Per-quantile posterior mean, standard deviation, median and
        credible bounds, plus the draws themselves.

    References
    ----------
    Kottas and Krnjajic (2009) Scand. J. Statist. 36(2), 297-319;
    Rubin (1981) Ann. Statist. 9(1), 130-134.
    """
    if route not in ROUTES:
        raise ValueError("route must be one of %r" % (ROUTES,))
    qs = ([float(quantile)] if not hasattr(quantile, "__len__")
          else [float(v) for v in quantile])
    if any(not (0.0 < v < 1.0) for v in qs):
        raise ValueError("quantiles must lie strictly inside (0, 1)")
    if not (0.0 < cred < 1.0):
        raise ValueError("cred must lie strictly inside (0, 1)")
    ys = [float(v) for v in y]
    n = len(ys)
    if n < 2:
        raise ValueError("need at least two observations")
    ybar = _w.csum(ys) / n
    sd = math.sqrt(_w.csum((v - ybar) * (v - ybar) for v in ys) / (n - 1))
    lo = min(ys) - 10.0 * sd
    hi = max(ys) + 10.0 * sd

    draws = {q: [] for q in qs}
    min_mass = 1.0
    fit = None

    if route == "bayesian_bootstrap":
        rng = _core._SplitMix64(seed)
        order = sorted(range(n), key=lambda i: (ys[i], i))
        ysort = [ys[i] for i in order]
        for _ in range(int(n_bootstrap)):
            # Dirichlet(1,...,1) as normalised unit exponentials, which
            # is Rubin's construction and needs no Dirichlet primitive.
            g = [float(rng.gamma(1.0, 1.0)) for _ in range(n)]
            tot = _w.csum(g)
            wts = [v / tot for v in g]
            for q in qs:
                draws[q].append(_weighted_quantile(ysort, wts, q))
    else:
        fit = slice_break_dp(ys, alpha=alpha, n_iter=n_iter, burn=burn,
                             thin=thin, route=sampler_route, kappa=kappa,
                             m0=m0, kappa0=kappa0, a0=a0, b0=b0,
                             seed=seed, alpha_update=alpha_update,
                             keep_draws=True)
        for d in fit["draws"]:
            mass = _w.csum(d["w"])
            if mass < min_mass:
                min_mass = mass
            if route == "mixture":
                for q in qs:
                    draws[q].append(mixture_quantile(q, d["w"], d["mu"],
                                                     d["s2"], None, None))
        if route == "predictive":
            # Average the CDF over sweeps, then invert ONCE. The order
            # matters: inversion does not commute with averaging, and
            # doing it the other way round would silently return the
            # mixture route's answer under a different name.
            sweeps = fit["draws"]
            m = len(sweeps)

            def fbar(x):
                return _w.csum(
                    mixture_cdf(x, d["w"], d["mu"], d["s2"])
                    / _w.csum(d["w"]) for d in sweeps) / m

            for q in qs:
                g = (lambda qq: (lambda x: fbar(x) - qq))(q)
                a, b = expand_bracket(g, lo, hi)
                draws[q].append(_w.bisect(g, a, b))

    out = []
    for q in qs:
        v = sorted(draws[q])
        m = len(v)
        mean = _w.csum(v) / m
        if m > 1:
            sdq = math.sqrt(_w.csum((t - mean) * (t - mean) for t in v)
                            / (m - 1))
        else:
            sdq = 0.0
        a = (1.0 - cred) / 2.0
        out.append({
            "q": q,
            "estimate": mean,
            "sd": sdq,
            "median": _weighted_quantile(v, [1.0 / m] * m, 0.5),
            "lower": _weighted_quantile(v, [1.0 / m] * m, a),
            "upper": _weighted_quantile(v, [1.0 / m] * m, 1.0 - a),
            "draws": draws[q],
        })

    payload = {
        "quantiles": out,
        "estimate": out[0]["estimate"],
        "se": out[0]["sd"],
        "n": n,
        "route": route,
        "cred": float(cred),
        "n_draws": len(draws[qs[0]]),
        "min_mass_carried": min_mass,
        "seed": int(seed),
        "method": "nonparametric Bayes posterior of the quantile function",
    }
    if fit is not None:
        payload["mean_clusters"] = fit["mean_clusters"]
        payload["sampler_route"] = fit["route"]
    return RichResult(payload=payload)


bnppct = bnp_percent_quantile


def cheatsheet():
    return ("bnppct: nonparametric Bayes posterior of the quantile "
            "function. routes " + ", ".join(ROUTES))
