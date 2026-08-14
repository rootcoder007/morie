# morie.fn -- function file (rootcoder007/morie)
r"""The targeted learning roadmap, as executable structure.

The roadmap is a sequence, and its order is the argument: (1) the data
are a realisation of a random variable with distribution :math:`P_0`;
(2) the **statistical model** :math:`\mathcal{M}` represents what is
genuinely known about the experiment that generated them -- no more;
(3) the scientific question becomes a **target parameter**
:math:`\Psi : \mathcal{M} \to \mathbb{R}`; (4) TMLE estimates it and
supplies inference.

**Why the order matters.** Fitting first and asking afterwards is what
produces an estimator biased for the question and non-normal in the
limit. The book's metaphor is exact: one cannot shoot the arrow and
then paint the bullseye -- the target must be specified in advance.

**A TMLE is three ingredients, and they are not independent.**

1. a target parameter :math:`\Psi` that is pathwise differentiable,
   with canonical gradient (efficient influence curve) :math:`D^*(P)`;
2. a **least favorable submodel** :math:`\{P(\epsilon)\}` through the
   initial estimator, used as an offset;
3. a **loss function** :math:`L` whose score along that submodel at
   :math:`\epsilon = 0` **spans** :math:`D^*`.

That last condition is the whole mechanism. Because the score spans
the efficient influence curve, the maximum likelihood step along the
submodel makes the updated estimator solve :math:`P_n D^*(P_n^*) = 0`,
and that equation is what delivers double robustness and asymptotic
efficiency of the substitution estimator. ``score_spans_eic`` checks
it numerically rather than taking it on faith -- if the submodel and
loss are mismatched, this is where it shows.

**Machine learning is used, but not trusted for inference.** The
initial fit should be a super learner: a cross-validated ensemble.
Its own bias is then removed by the targeting step, which is why an
estimator built on flexible learning can still be asymptotically
linear -- and why the roadmap does not require, or want, a
pre-specified parametric model.

References
----------
van der Laan, M. J. & Rose, S. (2018) *Targeted Learning in Data
Science: Causal Inference for Complex Longitudinal Studies*, Springer,
doi:10.1007/978-3-319-65304-4. Chap. 1 (the roadmap: data as a random
variable; a statistical model representing true knowledge of the
experiment; translation of the scientific question into a statistical
target parameter; TMLE with inference; the three requirements of a
TMLE -- a target parameter defined as a mapping from an infinite
dimensional parameter, a least favorable submodel through the initial
estimator, and a loss function whose score condition on the submodel
spans the efficient score, so the resulting substitution estimator
solves the efficient score equation, giving double robustness and
asymptotic efficiency; the use of a super learner for the initial fit;
and the warning that an untargeted fit is overly biased and not
normally distributed).

van der Laan, M. J. & Rose, S. (2011) *Targeted Learning: Causal
Inference for Observational and Experimental Data*, Springer,
doi:10.1007/978-1-4419-9782-1. The first book, which this one is a
sequel to.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["roadmap", "eic_ate", "score_spans_eic",
           "solves_eic_equation", "plugin"]

_EPS = 1e-12
_STEPS = ("data", "model", "target", "estimate")


def roadmap(data_description, model_assumptions, target_name,
            estimator="TMLE"):
    r"""Record the four steps in order, refusing an empty model.

    An empty ``model_assumptions`` means every assumption was left
    implicit, which is the failure the roadmap exists to prevent.
    """
    if not str(data_description).strip():
        raise ValueError("tlroad: the data-generating experiment must "
                         "be described")
    if not model_assumptions:
        raise ValueError("tlroad: the statistical model must state "
                         "what is known; an unstated model is a "
                         "parametric assumption you have not admitted")
    if not str(target_name).strip():
        raise ValueError("tlroad: the target parameter must be "
                         "specified BEFORE estimation")
    return {"steps": list(_STEPS), "data": str(data_description),
            "model": list(model_assumptions),
            "target": str(target_name), "estimator": str(estimator),
            "note": "the target is specified first: one cannot shoot "
                    "the arrow and then paint the bullseye"}


def eic_ate(A, Y, Q1, Q0, g, psi):
    r"""The efficient influence curve of the ATE.

    :math:`D^*(O) = \big(\frac{A}{g} - \frac{1-A}{1-g}\big)
    (Y - \bar Q_A) + \bar Q_1 - \bar Q_0 - \psi`.
    """
    a = [float(v) for v in k.vec(A)]
    y = [float(v) for v in k.vec(Y)]
    q1 = [float(v) for v in k.vec(Q1)]
    q0 = [float(v) for v in k.vec(Q0)]
    gg = [float(v) for v in k.vec(g)]
    n = len(a)
    if not (len(y) == len(q1) == len(q0) == len(gg) == n):
        raise ValueError("tlroad: the inputs differ in length")
    if any(v <= 0.0 or v >= 1.0 for v in gg):
        raise ValueError("tlroad: the propensity score must lie "
                         "strictly inside (0,1) -- a positivity "
                         "violation")
    out = []
    for i in range(n):
        qa = q1[i] if a[i] == 1.0 else q0[i]
        h = a[i] / gg[i] - (1.0 - a[i]) / (1.0 - gg[i])
        out.append(h * (y[i] - qa) + q1[i] - q0[i] - float(psi))
    return out


def score_spans_eic(A, Y, Q1, Q0, g, h=1e-6):
    r"""Check requirement 3: the loss's score spans :math:`D^*`.

    Fluctuates the initial fit through the logistic submodel with the
    clever covariate as offset, differentiates the log-likelihood loss
    at :math:`\epsilon = 0` numerically, and compares with the
    :math:`Y - \bar Q` part of the efficient influence curve. A
    mismatched loss and submodel fail here.
    """
    a = [float(v) for v in k.vec(A)]
    y = [float(v) for v in k.vec(Y)]
    q1 = [float(v) for v in k.vec(Q1)]
    q0 = [float(v) for v in k.vec(Q0)]
    gg = [float(v) for v in k.vec(g)]
    n = len(a)

    def loss(eps):
        tot = 0.0
        for i in range(n):
            qa = q1[i] if a[i] == 1.0 else q0[i]
            cc = a[i] / gg[i] - (1.0 - a[i]) / (1.0 - gg[i])
            lo = math.log(qa / (1.0 - qa)) + eps * cc
            p = 1.0 / (1.0 + math.exp(-lo))
            p = min(max(p, _EPS), 1.0 - _EPS)
            tot += -(y[i] * math.log(p)
                     + (1.0 - y[i]) * math.log(1.0 - p))
        return tot / n

    score = -(loss(h) - loss(-h)) / (2.0 * h)
    direct = 0.0
    for i in range(n):
        qa = q1[i] if a[i] == 1.0 else q0[i]
        cc = a[i] / gg[i] - (1.0 - a[i]) / (1.0 - gg[i])
        direct += cc * (y[i] - qa)
    direct /= n
    return {"score": score, "eic_component": direct,
            "difference": abs(score - direct),
            "spans": abs(score - direct) < 1e-5,
            "note": "the score of the loss along the submodel at "
                    "epsilon = 0 IS the efficient influence curve "
                    "component; that is what makes the update solve "
                    "the efficient score equation"}


def plugin(Q1, Q0):
    r"""The substitution estimator :math:`\Psi(Q) = E[\bar Q_1 -
    \bar Q_0]`."""
    q1 = [float(v) for v in k.vec(Q1)]
    q0 = [float(v) for v in k.vec(Q0)]
    if len(q1) != len(q0):
        raise ValueError("tlroad: the two arms differ in length")
    return sum(q1[i] - q0[i] for i in range(len(q1))) / len(q1)


def solves_eic_equation(A, Y, Q1, Q0, g, psi, tol=1e-8):
    r""":math:`P_n D^*(P_n^*) = 0` -- the equation targeting exists to
    solve."""
    d = eic_ate(A, Y, Q1, Q0, g, psi)
    m = sum(d) / len(d)
    se = math.sqrt(sum((v - m) ** 2 for v in d) / len(d) ** 2)
    return RichResult(payload={
        "estimate": m, "mean_eic": m, "solved": abs(m) < float(tol),
        "se": se, "ci": (float(psi) - 1.96 * se,
                         float(psi) + 1.96 * se),
        "method": "efficient score equation and influence-curve "
                  "inference; van der Laan & Rose (2018) Chap. 1",
    })


def cheatsheet():
    return ("tlroad: (1) data as a random variable, (2) a statistical "
            "model stating only what is KNOWN, (3) the scientific "
            "question as a target parameter, (4) TMLE plus inference "
            "-- in that order, because you cannot shoot the arrow then "
            "paint the bullseye. A TMLE needs three matched pieces: a "
            "pathwise differentiable parameter with canonical gradient "
            "D*, a least favorable submodel through the initial fit, "
            "and a LOSS WHOSE SCORE SPANS D*. That span is the "
            "mechanism: it makes the update solve P_n D* = 0, which is "
            "where double robustness and efficiency come from.")


# compact alias per ledger/NAMING.md
targetedroadmap = roadmap
