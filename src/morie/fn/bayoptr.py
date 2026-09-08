# morie.fn -- function file (rootcoder007/morie)
r"""Bayesian optimisation, chosen by acquisition function.

Mockus, J. (1975) "On Bayesian methods for seeking the extremum", in
*Optimization Techniques IFIP Technical Conference*, 400-404.

Snoek, J., Larochelle, H., & Adams, R. P. (2012) "Practical Bayesian
Optimization of Machine Learning Algorithms", *NIPS 25*.
arXiv:1206.2944

This is the same method as :mod:`morie.fn.bayopt` and the same two
papers; what it adds is a front end organised around the *choice of
acquisition function*, which is the decision Section 2 of Snoek et al.
spends its length on. The Gaussian process, the kernels and the closed
forms all live in :mod:`morie.fn.bayopt` and are imported, not copied,
so the two cannot drift apart.

``acquisition`` accepts the paper's names as well as the short codes:

============  ===========================================
``"ei"``      expected improvement, Equation 2 (default)
``"pi"``      probability of improvement, Equation 1
``"ucb"``     the confidence-bound rule of Equation 3
``"lcb"``     the same thing; the paper writes the *lower*
              bound because it minimises, and notes
              "upper, when considering maximization"
============  ===========================================

``"ucb"`` and ``"lcb"`` are one rule under two names, not two rules --
which is worth being explicit about, since a caller who asks for UCB on
a minimisation problem and silently gets a different criterion would
have no way to tell.

Everything else -- bounds, kernel, ``kappa``, ``xi``, the initial design
-- is passed straight through.
"""

from ._richresult import RichResult  # noqa: F401  (re-exported shape)
from .bayopt import (acquire, expected_improvement, gp_posterior,
                     lower_confidence_bound,
                     probability_of_improvement)
from .bayopt import bayopt as _bayopt

__all__ = [
    "bayoptr",
    "bayesian_optimization_ei_ucb",
    "resolve_acquisition",
    "ACQUISITIONS",
    "acquire",
    "gp_posterior",
    "expected_improvement",
    "probability_of_improvement",
    "lower_confidence_bound",
]

#: Accepted names, and the rule each resolves to.
ACQUISITIONS = {
    "ei": "ei",
    "expected_improvement": "ei",
    "pi": "pi",
    "probability_of_improvement": "pi",
    "ucb": "lcb",
    "lcb": "lcb",
    "confidence_bound": "lcb",
}


def resolve_acquisition(name):
    """Map a spelling to one of the three rules of Equations 1-3."""
    key = str(name).lower()
    if key not in ACQUISITIONS:
        raise ValueError("bayoptr: acquisition must be one of %s"
                         % (sorted(ACQUISITIONS),))
    return ACQUISITIONS[key]


def bayoptr(f, bounds, acquisition="ei", n_iter=20, n_init=5,
            kernel="matern52", amplitude=1.0, length_scale=1.0,
            noise=1e-8, kappa=2.0, xi=0.0, n_candidates=200, seed=0,
            X0=None, y0=None):
    """Minimise ``f``, choosing the acquisition function by name."""
    acq = resolve_acquisition(acquisition)
    res = _bayopt(f, bounds, n_iter=n_iter, n_init=n_init, acq=acq,
                  kernel=kernel, amplitude=amplitude,
                  length_scale=length_scale, noise=noise, kappa=kappa,
                  xi=xi, n_candidates=n_candidates, seed=seed, X0=X0,
                  y0=y0)
    payload = dict(res.payload)
    payload["acquisition"] = str(acquisition).lower()
    payload["acq"] = acq
    payload["note"] = (payload["note"] +
                       "; 'ucb' and 'lcb' name the same rule -- the "
                       "paper writes the lower bound because it "
                       "minimises, and says 'upper, when considering "
                       "maximization'")
    return RichResult(payload=payload)


bayesian_optimization_ei_ucb = bayoptr


def cheatsheet():
    return ("bayoptr: Bayesian optimisation chosen by acquisition "
            "(Mockus 1975; Snoek et al. 2012). A front end over "
            "morie.fn.bayopt that takes 'ei'/'expected_improvement', "
            "'pi'/'probability_of_improvement' or 'ucb'/'lcb' -- the "
            "last two being one rule under two names, since the paper "
            "writes the lower bound for minimisation. The GP and the "
            "closed forms are imported from bayopt, not duplicated.")
