# morie.fn -- function file (rootcoder007/morie)
r"""Particle filtering for partially observed Markov processes.

A POMP is specified by three things and nothing else: a simulator for
the latent process, a density for the measurement, and a simulator for
the initial state. The filter never needs the transition *density* --
only the ability to draw from it -- which is what makes the class so
broad, and it is the property this module is organised around.

**Plug-and-play.** Because only a simulator is required, the process
may be a stochastic differential equation, a compartmental epidemic
model, or anything else that can be stepped forward. The anchor uses
that: it runs the same filter over a model whose transition density is
never written down anywhere.

**Replicated filtering, because one run is a random variable.** The
likelihood estimate is unbiased but noisy, so pomp's practice is to
replicate the filter and combine. The combination is not a mean of the
log-likelihoods -- that would compound the downward Jensen bias -- but
a log of the mean likelihood, computed by log-sum-exp so it does not
underflow. Both are returned, because their difference *is* the Monte
Carlo error, and a shrinking gap is the evidence that enough particles
were used.

**The profile is over the estimate, not the truth.** Evaluating the
likelihood on a grid of a parameter gives a profile whose maximiser
estimates the MLE, but each point carries its own filtering noise, so
a profile drawn with too few particles is rough in a way that looks
like structure. The standard error across replicates is reported at
every grid point for that reason.

References
----------
King, A. A., Nguyen, D. & Ionides, E. L. (2016) "Statistical Inference
for Partially Observed Markov Processes: The R Package pomp", *Journal
of Statistical Software* 69(12), 1-43, doi:10.18637/jss.v069.i12.
Algorithms 1-2, the plug-and-play property, and replicated filtering.

Ionides, E. L., Nguyen, D., Atchade, Y., Stoev, S. & King, A. A. (2015)
"Inference for dynamic and latent variable models via iterated,
perturbed Bayes maps", *Proceedings of the National Academy of
Sciences* 112(3), 719-724, doi:10.1073/pnas.1410597112. Iterated
filtering, the maximisation this supports.

Bretó, C., He, D., Ionides, E. L. & King, A. A. (2009) "Time series
analysis via mechanistic models", *The Annals of Applied Statistics*
3(1), 319-348, doi:10.1214/08-AOAS201. The plug-and-play framing.

Andrieu, C., Doucet, A. & Holenstein, R. (2010) "Particle Markov chain
Monte Carlo methods", *Journal of the Royal Statistical Society Series
B* 72(3), 269-342, doi:10.1111/j.1467-9868.2009.00736.x. Why the
unbiasedness of the likelihood estimate matters.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult
from .prtcl import particle_filter

__all__ = ["replicated_pfilter", "loglik_profile", "logmeanexp"]

_EPS = 1e-300


def logmeanexp(values):
    r""":math:`\log\frac1n\sum e^{v_i}`, without underflowing.

    Averaging the LIKELIHOODS and then taking the log -- not averaging
    the log-likelihoods, which compounds the downward Jensen bias the
    single-run estimate already carries.
    """
    v = [float(x) for x in values]
    if not v:
        raise ValueError("pftrep: nothing to average")
    mx = max(v)
    if mx == float("-inf"):
        return float("-inf")
    return mx + math.log(sum(math.exp(x - mx) for x in v) / len(v))


def replicated_pfilter(y, n_particles, init, step, loglik,
                       n_reps=10, seed=0, **kw):
    r"""Run the filter several times and combine on the right scale.

    Returns both the log of the mean likelihood and the mean of the
    log-likelihoods; their gap IS the Monte Carlo error, and it should
    shrink as particles are added.
    """
    R = int(n_reps)
    if R < 1:
        raise ValueError("pftrep: need at least 1 replicate, got %d" % R)
    lls, minless = [], []
    for r in range(R):
        res = particle_filter(y, n_particles, init, step, loglik,
                              seed=seed * 1013 + r, **kw)
        lls.append(res["loglik"])
        minless.append(res["min_ess"])
    lme = logmeanexp(lls)
    mean_ll = sum(lls) / R
    se = (k.sd(lls) / math.sqrt(R)) if R > 1 else float("nan")
    return RichResult(payload={
        "estimate": lme, "loglik": lme, "logmeanexp": lme,
        "mean_loglik": mean_ll, "jensen_gap": lme - mean_ll,
        "se": se, "replicates": lls, "n_reps": R,
        "n_particles": int(n_particles),
        "min_ess": min(minless), "mean_min_ess": sum(minless) / R,
        "method": "replicated particle filtering, King, Nguyen & "
                  "Ionides (2016)",
    })


def loglik_profile(y, grid, make_model, n_particles=200, n_reps=5,
                   seed=0, **kw):
    r"""Profile the likelihood over a parameter grid.

    ``make_model(theta) -> (init, step, loglik)``. Each grid point
    carries its own filtering noise, so the standard error is returned
    beside every value -- a profile drawn with too few particles is
    rough in a way that reads as structure.
    """
    g = [float(v) for v in grid]
    if len(g) < 2:
        raise ValueError("pftrep: need at least 2 grid points, got %d"
                         % len(g))
    vals, ses = [], []
    for t, th in enumerate(g):
        init, step, loglik = make_model(th)
        r = replicated_pfilter(y, n_particles, init, step, loglik,
                               n_reps=n_reps, seed=seed + 97 * t, **kw)
        vals.append(r["loglik"])
        ses.append(r["se"])
    best = max(range(len(g)), key=lambda i: vals[i])
    return RichResult(payload={
        "estimate": g[best], "mle": g[best], "grid": g,
        "loglik": vals, "se": ses, "max_loglik": vals[best],
        "n_particles": int(n_particles), "n_reps": int(n_reps),
        "method": "particle-filter likelihood profile, King, Nguyen & "
                  "Ionides (2016)",
    })


def cheatsheet():
    return ("pftrep: a POMP needs only a SIMULATOR for the latent "
            "process plus a measurement density -- never the transition "
            "density, which is what plug-and-play means. Replicate the "
            "filter and combine with logmeanexp (log of the MEAN "
            "likelihood), not the mean of the logs, which compounds the "
            "Jensen bias. The gap between them IS the Monte Carlo "
            "error.")


# compact alias per ledger/NAMING.md
replicatedpfilter = replicated_pfilter
