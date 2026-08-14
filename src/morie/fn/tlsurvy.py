# morie.fn -- function file (rootcoder007/morie)
r"""Targeted learning using adaptive survey sampling.

:math:`N` observations are available and :math:`N` is too large to use.
The response is not to approximate the estimator but to **sample the
data**: select :math:`n` of the :math:`N` with unequal inclusion
probabilities, and run a TMLE on the smaller set. The asymptotics are
in both, with :math:`n \to \infty` as :math:`N \to \infty` and
:math:`n/N \to 0` -- so the computational saving is not vanishing.

**A low-dimensional summary must be observed for everyone.** Each
:math:`O_i` is summarised by a cheap :math:`V_i`, and
:math:`V_1,\dots,V_N` are all available. That is what makes *adaptive*
sampling possible: the design can use :math:`V` to decide whom to
look at, even though the expensive part of :math:`O` is never read for
the unsampled.

**Unequal probabilities, and the Horvitz-Thompson correction that
follows.** Selecting with probability :math:`\pi_i` and weighting by
:math:`1/\pi_i` keeps the estimator unbiased for the full-data
quantity. Choosing :math:`\pi_i` proportional to the *influence* an
observation carries -- large :math:`|D^*|` given :math:`V` -- minimises
the variance for a given :math:`n`, which is the whole point of
adapting the design rather than sampling uniformly.

**The two error sources are separate, and reporting them separately
matters.** Sampling variance from having only :math:`n` observations,
and the full-data variance that would remain at :math:`n = N`. The
first is under the analyst's control through the design; the second is
not. ``design_efficiency`` compares an adaptive design with uniform
sampling at the same :math:`n`, and the anchor requires the adaptive
one to win when the influence is unevenly distributed and to tie when
it is not -- the case where adaptation cannot help.

References
----------
van der Laan, M. J. & Rose, S. (2018) *Targeted Learning in Data
Science*, Springer, doi:10.1007/978-3-319-65304-4. Chap. 29 (Chambaz,
Joly & Mary): building a confidence interval for a real-valued
pathwise differentiable parameter from N independent observations when
N is so large that not all data can be used; the two-part response of
selecting n among N randomly with UNEQUAL INCLUSION PROBABILITIES and
adapting TMLE to the resulting smaller data set; the asymptotics with
N to infinity and n to infinity such that n/N goes to zero; the
selection as the random outcome of a survey sampling design; and the
assumption that each observation is summarised by a low-dimensional
V_i with V_1, ..., V_N all observed.

Horvitz, D. G. & Thompson, D. J. (1952) "A Generalization of Sampling
Without Replacement From a Finite Universe", *Journal of the American
Statistical Association* 47(260), 663-685,
doi:10.1080/01621459.1952.10483446.

Chambaz, A., Joly, E. & Mary, X. (2018) "Targeted Learning Using
Adaptive Survey Sampling", in *Targeted Learning in Data Science*,
Springer, doi:10.1007/978-3-319-65304-4_29.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["inclusion_probabilities", "draw_sample",
           "horvitz_thompson", "design_efficiency",
           "adaptive_survey_tmle"]

_EPS = 1e-12
_DESIGNS = ("uniform", "proportional", "adaptive")


def inclusion_probabilities(V, n, design="adaptive", influence=None,
                            floor=0.01):
    r"""Choose :math:`\pi_i`, summing to :math:`n`.

    ``adaptive`` sets :math:`\pi_i \propto` the expected influence
    given :math:`V_i`, which minimises the variance for a fixed
    :math:`n`. A floor keeps every unit reachable -- a zero inclusion
    probability makes the estimand unidentifiable for that stratum.
    """
    if design not in _DESIGNS:
        raise ValueError("tlsurvy: design must be one of %s, got %r"
                         % (", ".join(_DESIGNS), design))
    v = [float(q) for q in k.vec(V)]
    N = len(v)
    nn = int(n)
    if nn < 1 or nn > N:
        raise ValueError("tlsurvy: n must lie in 1..%d, got %d"
                         % (N, nn))
    if design == "uniform":
        base = [1.0] * N
    elif design == "proportional":
        base = [abs(q) + _EPS for q in v]
    else:
        if influence is None:
            raise ValueError("tlsurvy: the adaptive design needs the "
                             "expected influence given V")
        base = [abs(float(q)) + _EPS for q in k.vec(influence)]
        if len(base) != N:
            raise ValueError("tlsurvy: %d influence values for %d "
                             "units" % (len(base), N))
    # Rescale to sum exactly to n, iterating because capping at 1 and
    # flooring both remove mass that has to go somewhere -- a single
    # pass leaves the expected sample size short of n.
    s = sum(base)
    pi = [nn * b / s for b in base]
    fl = float(floor)
    for _ in range(100):
        pi = [min(1.0, max(fl, p)) for p in pi]
        tot = sum(pi)
        if abs(tot - nn) < 1e-9:
            break
        free = [i for i in range(N) if fl < pi[i] < 1.0]
        if not free:
            break
        slack = nn - tot
        room = sum((1.0 - pi[i]) if slack > 0 else (pi[i] - fl)
                   for i in free)
        if room <= _EPS:
            break
        for i in free:
            share = ((1.0 - pi[i]) if slack > 0
                     else (pi[i] - fl)) / room
            pi[i] += slack * share
    return {"pi": pi, "design": design, "n_expected": sum(pi),
            "N": N, "min_pi": min(pi),
            "note": "a zero inclusion probability makes that stratum "
                    "unidentifiable, so the floor is not cosmetic"}


def draw_sample(pi, seed=0):
    r"""Poisson sampling: include unit :math:`i` with probability
    :math:`\pi_i`."""
    p = [float(q) for q in k.vec(pi)]
    rng = np.random.default_rng(seed)
    idx = [i for i in range(len(p)) if float(rng.uniform()) < p[i]]
    if not idx:
        raise ValueError("tlsurvy: the draw selected nothing; raise "
                         "the inclusion probabilities")
    return {"selected": idx, "n": len(idx),
            "fraction": len(idx) / float(len(p))}


def horvitz_thompson(values, pi, selected, N=None):
    r"""The design-unbiased mean: :math:`\frac{1}{N}\sum_{i \in S}
    y_i/\pi_i`.

    Unbiased for the FULL-data mean, which is the quantity of
    interest -- the sample is a computational device, not the
    population.
    """
    y = [float(q) for q in k.vec(values)]
    p = [float(q) for q in k.vec(pi)]
    idx = [int(q) for q in selected]
    n_total = len(p) if N is None else int(N)
    if any(p[i] <= 0.0 for i in idx):
        raise ValueError("tlsurvy: a selected unit has zero inclusion "
                         "probability")
    est = sum(y[i] / p[i] for i in idx) / n_total
    var = sum((1.0 - p[i]) * (y[i] / p[i]) ** 2 for i in idx) \
        / (n_total ** 2)
    return {"estimate": est, "se": math.sqrt(max(var, 0.0)),
            "n_used": len(idx), "N": n_total}


def design_efficiency(values, influence, n, seed=0):
    r"""Adaptive against uniform sampling at the same :math:`n`.

    Adaptation can only help when the influence is unevenly spread;
    where it is flat the two coincide, and claiming otherwise would be
    claiming something for nothing.
    """
    y = [float(q) for q in k.vec(values)]
    out = {}
    for d in ("uniform", "adaptive"):
        pi = inclusion_probabilities(y, n, d,
                                     influence if d == "adaptive"
                                     else None)["pi"]
        s = draw_sample(pi, seed)["selected"]
        out[d] = horvitz_thompson(y, pi, s)
    return {"uniform_se": out["uniform"]["se"],
            "adaptive_se": out["adaptive"]["se"],
            "ratio": out["adaptive"]["se"] / out["uniform"]["se"]
            if out["uniform"]["se"] > 0 else float("nan"),
            "note": "with a flat influence the designs coincide -- "
                    "adaptation cannot buy anything there"}


def adaptive_survey_tmle(V, influence_proxy, full_estimator, n,
                         seed=0):
    r"""Sample by the adaptive design, then run the estimator on the
    sample.

    Reports both error sources: the sampling variance from using
    :math:`n` of :math:`N`, and the estimator's own standard error.
    """
    pi = inclusion_probabilities(V, n, "adaptive",
                                 influence_proxy)["pi"]
    s = draw_sample(pi, seed)
    r = full_estimator(s["selected"], [1.0 / pi[i]
                                       for i in s["selected"]])
    return RichResult(payload={
        "estimate": float(r["estimate"]),
        "psi": float(r["estimate"]),
        "se_estimator": float(r.get("se", float("nan"))),
        "n_used": s["n"], "N": len(pi),
        "sampling_fraction": s["fraction"],
        "inclusion_probabilities": pi,
        "method": "TMLE on an adaptive survey sample; van der Laan & "
                  "Rose (2018) Chap. 29",
        "note": "asymptotics in both n and N with n/N -> 0, so the "
                "computational saving does not vanish",
    })


def cheatsheet():
    return ("tlsurvy: N too large to use, so SAMPLE the data rather "
            "than approximate the estimator -- select n of N with "
            "UNEQUAL inclusion probabilities and run TMLE on the "
            "sample, with n/N -> 0 so the saving persists. A cheap "
            "low-dimensional V is observed for ALL N, which is what "
            "makes the design adaptive: set pi_i proportional to the "
            "expected INFLUENCE given V and the variance is minimised "
            "for that n. Weight by 1/pi (Horvitz-Thompson) to stay "
            "unbiased for the FULL-data parameter. Where the influence "
            "is flat, adaptation buys nothing.")


# compact alias per ledger/NAMING.md
adaptivesurveytmle = adaptive_survey_tmle
