# morie.fn -- function file (rootcoder007/morie)
r"""Defining the model and parameter: the g-computation estimand.

A causal question is a question about a distribution we do not observe.
The structural causal model generates :math:`(U, X)`; the observed data
:math:`O` are a function of :math:`X`; and the causal target parameter
:math:`\Psi^F(P_{U,X})` -- a counterfactual mean under an intervention
rule -- is a parameter of the *full* data distribution.

**Identification is what makes it estimable.** Under sequential
randomization and positivity, that causal quantity equals a functional
of the observed data distribution, the **g-computation formula**: for
a longitudinal structure,

.. math:: \Psi(P) = \sum_{\bar l}\ E\big[Y \mid \bar A = \bar a,
          \bar L = \bar l\big]\prod_{t}
          P\big(L_t = l_t \mid \bar A_{t-1} = \bar a_{t-1},
          \bar L_{t-1} = \bar l_{t-1}\big),

which integrates the outcome regression over the covariate law while
*holding the intervention fixed*. The treatment mechanism does not
appear, which is exactly the point: the intervention replaces it.

**Two assumptions, doing different jobs.** Sequential randomization
says treatment at each time is independent of the counterfactuals
given the observed past -- no unmeasured confounding. Positivity says
every treatment history retains positive probability given the past;
without it, the formula asks the outcome regression to extrapolate
into cells that contain no data. The anchor breaks each separately and
shows the estimand moves.

**And a point the book insists on.** Whether or not the causal
interpretation survives, :math:`\Psi(P)` remains a well-defined
statistical parameter with a valid statistical interpretation. The
estimation problem is defined by the model and the estimand, not by
the causal story.

References
----------
van der Laan, M. J. & Rose, S. (2018) *Targeted Learning in Data
Science*, Springer, doi:10.1007/978-3-319-65304-4. Chap. 2 (the causal
model for longitudinal data; the causal target parameter
Psi^F(P_{U,X}) defined by counterfactual means under an intervention
rule; identification of that quantity as a function of the observed
data distribution by the g-computation estimand under sequential
randomization and positivity; and the point that under those
assumptions the estimand equals the causal quantity, but either way it
has a valid statistical interpretation). Chap. 4 (the g-computation
formula in the running longitudinal example).

Robins, J. M. (1986) "A new approach to causal inference in mortality
studies with a sustained exposure period", *Mathematical Modelling*
7(9-12), 1393-1512, doi:10.1016/0270-0255(86)90088-6. The
g-computation formula.

Pearl, J. (2009) *Causality: Models, Reasoning, and Inference*, 2nd
edition, Cambridge University Press,
doi:10.1017/CBO9780511803161. Causal graphs and the backdoor
criterion, discussed in Chap. 2 as an alternative route to
identifiability.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["g_computation", "positivity_check",
           "sequential_g_formula", "counterfactual_mean"]

_EPS = 1e-12


def positivity_check(g, delta=0.01):
    r"""Every intervention must retain positive probability.

    Returns the worst propensity rather than only a verdict: how
    close to the boundary the data get is the practically useful
    number.
    """
    gg = [float(v) for v in k.vec(g)]
    if not gg:
        raise ValueError("tlgcmp: no propensity scores given")
    lo, hi = min(gg), max(gg)
    worst = min(lo, 1.0 - hi)
    return {"min_g": lo, "max_g": hi, "worst": worst,
            "satisfied": worst > float(delta), "delta": float(delta),
            "note": "without positivity the outcome regression is "
                    "asked to extrapolate into cells with no data"}


def g_computation(strata, outcome_means, covariate_probs):
    r"""The point-treatment g-formula: average the regression over the
    covariate law with treatment held fixed.

    ``outcome_means[l]`` is :math:`E[Y \mid A = a, L = l]` and
    ``covariate_probs[l]` is :math:`P(L = l)`.
    """
    s = list(strata)
    p = [float(covariate_probs[v]) for v in s]
    if abs(sum(p) - 1.0) > 1e-9:
        raise ValueError("tlgcmp: the covariate distribution must sum "
                         "to 1, got %.9f" % sum(p))
    q = [float(outcome_means[v]) for v in s]
    return sum(p[i] * q[i] for i in range(len(s)))


def sequential_g_formula(Q_functions, L_supports, L_probs, rule):
    r"""The longitudinal g-formula by backward recursion.

    ``Q_functions[t](history)`` gives
    :math:`E[Q_{t+1} \mid \bar A_t = \bar a_t, \bar L_t]`, and the
    recursion integrates the covariate law forward while the
    intervention rule fixes treatment -- which is why the treatment
    mechanism never enters.
    """
    T = len(L_supports)
    if len(L_probs) != T:
        raise ValueError("tlgcmp: %d covariate supports but %d "
                         "distributions" % (T, len(L_probs)))

    def walk(t, hist):
        if t == T:
            return Q_functions[T](hist)
        tot = 0.0
        probs = L_probs[t](hist)
        if abs(sum(probs) - 1.0) > 1e-9:
            raise ValueError("tlgcmp: the conditional law at time %d "
                             "sums to %.9f" % (t, sum(probs)))
        for j, l in enumerate(L_supports[t]):
            a = rule(hist + [l])
            tot += probs[j] * walk(t + 1, hist + [l, a])
        return tot

    val = walk(0, [])
    return RichResult(payload={
        "estimate": val, "psi": val, "horizon": T,
        "method": "sequential g-computation; van der Laan & Rose "
                  "(2018) Chaps. 2 and 4",
        "note": "the treatment mechanism does not appear -- the "
                "intervention replaces it",
        "assumptions": ("sequential randomization (no unmeasured "
                        "confounding) and positivity"),
    })


def counterfactual_mean(Y, A, L, a_star, strata_probs=None):
    r"""The stratified estimate of :math:`E[Y_{a^*}]`.

    Averages the within-stratum mean under :math:`A = a^*` over the
    covariate distribution -- and refuses when a stratum contains no
    treated (or untreated) unit, because that is a positivity
    violation rather than a missing number.
    """
    y = [float(v) for v in k.vec(Y)]
    a = [float(v) for v in k.vec(A)]
    l = [int(v) for v in k.vec(L)]
    if not (len(y) == len(a) == len(l)):
        raise ValueError("tlgcmp: the inputs differ in length")
    levels = sorted(set(l))
    if strata_probs is None:
        strata_probs = {v: sum(1 for x in l if x == v) / float(len(l))
                        for v in levels}
    tot = 0.0
    for v in levels:
        idx = [i for i in range(len(y))
               if l[i] == v and a[i] == float(a_star)]
        if not idx:
            raise ValueError("tlgcmp: stratum %r contains no unit "
                             "with A = %r -- a positivity violation, "
                             "not a missing value" % (v, a_star))
        tot += strata_probs[v] * sum(y[i] for i in idx) / len(idx)
    return tot


def cheatsheet():
    return ("tlgcmp: the causal parameter lives on the FULL data "
            "(U, X); identification maps it to a functional of the "
            "OBSERVED data. The g-computation formula integrates the "
            "outcome regression over the covariate law with treatment "
            "held FIXED, so the treatment mechanism disappears -- the "
            "intervention replaced it. Two assumptions doing different "
            "jobs: sequential randomization (no unmeasured "
            "confounding) and positivity (every history keeps positive "
            "probability). Break positivity and the regression is "
            "asked to extrapolate into empty cells. Either way "
            "Psi(P) remains a valid STATISTICAL parameter.")


# compact alias per ledger/NAMING.md
gcomputation = sequential_g_formula
