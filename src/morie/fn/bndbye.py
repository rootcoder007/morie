# morie.fn -- function file (rootcoder007/morie)
r"""Bayesian credible sets vs frequentist confidence sets under
partial identification.

In a well-behaved problem the two coincide asymptotically -- a 95%
interval is a 95% interval, and a Bayesian reader of a frequentist
paper can use the numbers and vice versa. **That equivalence breaks
down when the parameter is only partially identified**, and it breaks
down in a specific, predictable direction.

**The structure.** The likelihood is indexed by a finite-dimensional,
*identifiable* reduced-form parameter :math:`\phi`. Given
:math:`\phi`, the structural parameter of interest is known only to lie
in the identified set :math:`\Theta(\phi)`. Data update beliefs about
:math:`\phi` through the likelihood, but -- the insight the paper
traces to Kadane -- the conditional distribution
:math:`\mathbb P_{\phi\theta}` of :math:`\theta` given :math:`\phi` is
**never updated**, because the likelihood does not depend on
:math:`\theta` given :math:`\phi`. It is whatever the prior said, for
ever.

**The consequence for credible sets.** As :math:`n` grows the
posterior of :math:`\phi` concentrates on :math:`\hat\phi_n`, so the
posterior of :math:`\theta` converges to the *conditional prior* at
:math:`\hat\phi_n`. A highest-posterior-density set is the smallest
set of a given probability, so unless that conditional prior is
uniform on :math:`\Theta(\hat\phi_n)` the HPD set **excludes parts of
the estimated identified set** -- it concentrates where the prior
happened to put mass.

**The consequence for confidence sets.** A frequentist set must cover
every :math:`\theta \in \Theta(\phi_0)` with the stated probability,
and :math:`\hat\phi_n` is estimated with error, so the set must
**extend beyond the boundaries** of :math:`\Theta(\hat\phi_n)`.

Put together: credible sets tend to be **smaller** than confidence
sets in large samples. That is not a paradox and neither procedure is
wrong -- they answer different questions -- but it explains why sign-
restricted VAR credible bands look narrow beside frequentist bands,
and why moment-inequality confidence sets look absurdly conservative
to a Bayesian.

**The normative point, which this module implements.** Report the
estimated identified set :math:`\Theta(\hat\phi_n)` **and** the
conditional prior alongside any credible set, because the credible set
alone cannot be interpreted without them. A prior approximately
uniform on the identified set is a useful benchmark. Every function
here returns all three.

References
----------
Moon, H. R. & Schorfheide, F. (2012) "Bayesian and Frequentist
Inference in Partially Identified Models", *Econometrica* 80(2),
755-782, doi:10.3982/ECTA8360. The abstract's three claims (posterior
HPD sets exclude parts of the estimated identified set, frequentist
sets extend beyond it, and both should be reported alongside the
conditional prior), the Kadane insight that P_{phi,theta} is not
updated by data, and the recommendation of a prior approximately
uniform on the identified set as a benchmark.

Imbens, G. W. & Manski, C. F. (2004) "Confidence Intervals for
Partially Identified Parameters", *Econometrica* 72(6), 1845-1857,
doi:10.1111/j.1468-0262.2004.00555.x. Frequentist inference for an
interval-identified parameter, cited in the paper's review of the
confidence-set literature and used here for the
``target="parameter"`` critical value.

Stoye, J. (2009) "More on Confidence Intervals for Partially
Identified Parameters", *Econometrica* 77(4), 1299-1315,
doi:10.3982/ECTA7347. The extension where length and location are
estimated at the same rate.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["identified_set_interval", "posterior_hpd",
           "frequentist_confidence_set", "compare_sets",
           "conditional_prior_uniform"]

_EPS = 1e-12


def identified_set_interval(phi_hat, half_width):
    r"""The plug-in estimate :math:`\Theta(\hat\phi_n)`.

    Kept deliberately simple -- an interval centred on the reduced-form
    estimate -- because the paper's results are about the *relation*
    between the three sets, not about any particular identified-set
    geometry.
    """
    h = float(half_width)
    if h < 0.0:
        raise ValueError("bndbye: the half-width must be "
                         "non-negative, got %r" % (half_width,))
    return {"lower": float(phi_hat) - h, "upper": float(phi_hat) + h,
            "width": 2.0 * h, "phi_hat": float(phi_hat)}


def conditional_prior_uniform(theta_set, n_grid=401):
    r"""The benchmark conditional prior: uniform on
    :math:`\Theta(\phi)`."""
    lo, hi = float(theta_set["lower"]), float(theta_set["upper"])
    if hi < lo:
        raise ValueError("bndbye: the identified set is empty")
    if hi - lo <= _EPS:
        return {"grid": [lo], "density": [1.0]}
    g = [lo + (hi - lo) * i / (int(n_grid) - 1)
         for i in range(int(n_grid))]
    return {"grid": g, "density": [1.0 / (hi - lo)] * len(g)}


def posterior_hpd(theta_set, level=0.95, conditional_prior=None,
                  n_grid=401):
    r"""The highest-posterior-density credible set, in the large-sample
    limit.

    In that limit the posterior of :math:`\theta` **is** the
    conditional prior at :math:`\hat\phi_n`, so the HPD set is the
    smallest region of that prior carrying ``level`` probability. With
    a uniform conditional prior it fills the identified set; with any
    non-uniform one it is strictly smaller.
    """
    if not 0.0 < float(level) < 1.0:
        raise ValueError("bndbye: level must lie in (0, 1)")
    cp = conditional_prior or conditional_prior_uniform(theta_set,
                                                        n_grid)
    g = [float(v) for v in cp["grid"]]
    d = [float(v) for v in cp["density"]]
    if len(g) != len(d):
        raise ValueError("bndbye: the prior grid and density differ "
                         "in length")
    if len(g) == 1:
        return {"lower": g[0], "upper": g[0], "width": 0.0,
                "level": float(level), "covered": 1.0}
    step = (g[-1] - g[0]) / (len(g) - 1)
    mass = [v * step for v in d]
    tot = sum(mass)
    if tot <= _EPS:
        raise ValueError("bndbye: the conditional prior has no mass")
    mass = [v / tot for v in mass]
    order = sorted(range(len(g)), key=lambda i: -d[i])
    acc, chosen = 0.0, []
    for i in order:
        chosen.append(i)
        acc += mass[i]
        if acc >= float(level):
            break
    lo = min(g[i] for i in chosen)
    hi = max(g[i] for i in chosen)
    return {"lower": lo, "upper": hi, "width": hi - lo,
            "level": float(level), "covered": acc,
            "n_grid_points": len(chosen),
            "method": "HPD of the conditional prior at phi_hat -- the "
                      "large-sample limit of the posterior "
                      "(Moon & Schorfheide 2012)"}


def frequentist_confidence_set(theta_set, se_phi, level=0.95,
                               target="parameter"):
    r"""A confidence set that must cover every point of the identified
    set.

    ``target="parameter"`` uses the Imbens-Manski critical value for
    covering the true :math:`\theta`; ``target="set"`` uses the
    two-sided value for covering the whole identified set, which is
    wider. Either way the set extends **beyond** the boundaries of
    :math:`\Theta(\hat\phi_n)`, because :math:`\hat\phi_n` is noisy.
    """
    if target not in ("parameter", "set"):
        raise ValueError("bndbye: target must be parameter or set, "
                         "got %r" % (target,))
    s = float(se_phi)
    if s < 0.0:
        raise ValueError("bndbye: the standard error must be "
                         "non-negative")
    if not 0.0 < float(level) < 1.0:
        raise ValueError("bndbye: level must lie in (0, 1)")
    c = (k.qnorm(float(level)) if target == "parameter"
         else k.qnorm(0.5 + float(level) / 2.0))
    return {"lower": theta_set["lower"] - c * s,
            "upper": theta_set["upper"] + c * s,
            "width": theta_set["width"] + 2.0 * c * s,
            "critical_value": c, "target": target,
            "level": float(level),
            "note": "extends beyond Theta(phi_hat) by c * se on each "
                    "side, because phi_hat is estimated"}


def compare_sets(phi_hat, half_width, se_phi, level=0.95,
                 conditional_prior=None, n_grid=401):
    r"""The paper's comparison, reported together as it recommends."""
    ts = identified_set_interval(phi_hat, half_width)
    hpd = posterior_hpd(ts, level=level,
                        conditional_prior=conditional_prior,
                        n_grid=n_grid)
    cs = frequentist_confidence_set(ts, se_phi, level=level)
    return RichResult(payload={
        "estimate": hpd["width"] / max(cs["width"], _EPS),
        "identified_set": ts, "credible_hpd": hpd,
        "confidence_set": cs,
        "hpd_inside_identified_set":
            hpd["lower"] >= ts["lower"] - 1e-9
            and hpd["upper"] <= ts["upper"] + 1e-9,
        "cs_contains_identified_set":
            cs["lower"] <= ts["lower"] + 1e-9
            and cs["upper"] >= ts["upper"] - 1e-9,
        "width_ratio_hpd_over_cs": hpd["width"] / max(cs["width"],
                                                      _EPS),
        "conditional_prior_reported": conditional_prior is not None,
        "method": "Moon & Schorfheide (2012): HPD excludes parts of "
                  "Theta(phi_hat); the confidence set extends beyond "
                  "it",
        "recommendation": "report Theta(phi_hat) and the conditional "
                          "prior alongside any credible set -- the "
                          "credible set alone cannot be interpreted",
    })


def cheatsheet():
    return ("bndbye: partial identification, Bayes vs frequentist. "
            "Data update phi through the likelihood, but the "
            "CONDITIONAL prior of theta given phi is NEVER updated "
            "(Kadane). So the posterior tends to the conditional "
            "prior at phi_hat, and the HPD set EXCLUDES parts of "
            "Theta(phi_hat) unless that prior is uniform. A "
            "confidence set must instead EXTEND BEYOND Theta(phi_hat) "
            "because phi_hat is noisy. Hence credible sets are "
            "SMALLER than confidence sets. Report the identified set "
            "and the conditional prior, not just the credible set.")


# compact alias per ledger/NAMING.md
bayescrediblebound = compare_sets

# public names resolved by fn/_lazy_map.json
bound_bayes_credible = compare_sets
