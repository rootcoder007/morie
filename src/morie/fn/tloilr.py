# morie.fn -- function file (rootcoder007/morie)
r"""Optimal individualised treatment under a resource constraint.

The unconstrained optimal rule treats everyone whose conditional
treatment effect ("blip")
:math:`B(W) = E[Y \mid A=1, W] - E[Y \mid A=0, W]` is positive. Real
programmes cannot: at most a proportion :math:`\kappa` of the
population can be treated.

**The constrained rule is a threshold on the blip.** Treat the units
with the largest blip until the budget is exhausted, i.e.
:math:`d_\kappa(W) = I\{B(W) > \tau_\kappa\}` where
:math:`\tau_\kappa` is the :math:`(1-\kappa)` quantile of :math:`B(W)`
-- and :math:`\tau_\kappa = 0` when the constraint does not bind, which
recovers the unconstrained rule exactly. The value is
:math:`E[Y_{d_\kappa}]`.

**The constraint makes estimation easier, not harder, and that is the
chapter's point.** Regular estimation of the *unconstrained* optimal
value requires a **nonexceptional law**: the blip must not have a
point mass at zero, because there the optimal rule is not uniquely
defined and the value is not pathwise differentiable. Under an
*active* constraint with continuous covariates the relevant condition
is instead about the blip's density at the threshold
:math:`\tau_\kappa > 0`, which is far more reasonable than assuming
nothing sits exactly at zero. So the constrained problem admits a
root-:math:`n` estimator with valid confidence intervals in settings
where the unconstrained one does not.

``exceptional_law`` reports the mass at zero for exactly this reason,
and the anchor uses a blip with a deliberate atom at zero to show the
unconstrained problem degrading while the constrained one does not.

References
----------
van der Laan, M. J. & Rose, S. (2018) *Targeted Learning in Data
Science*, Springer, doi:10.1007/978-3-319-65304-4. Chap. 23 (Luedtke &
van der Laan): a resource constraint under which a maximum proportion
of the population can be treated; a root-n rate estimator for the
optimal resource-constrained value with confidence intervals;
efficiency among all regular asymptotically linear estimators in the
nonparametric model; and the statement that when the baseline
covariates are continuous and the resource constraint is ACTIVE -- the
constrained value strictly below the unconstrained one -- the
conditions are more reasonable than the nonexceptional law assumption
needed for regular estimation of the optimal unconstrained value in
Chap. 22. The data structure (W, A, Y) with Y bounded in the unit
interval, noting any bounded continuous outcome can be rescaled.

Luedtke, A. R. & van der Laan, M. J. (2016) "Optimal Individualized
Treatments in Resource-Limited Settings", *International Journal of
Biostatistics* 12(1), 283-303, doi:10.1515/ijb-2015-0007.

Luedtke, A. R. & van der Laan, M. J. (2016) "Statistical inference
for the mean outcome under a possibly non-unique optimal treatment
strategy", *Annals of Statistics* 44(2), 713-742,
doi:10.1214/15-AOS1384. The nonexceptional law condition.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["blip", "resource_threshold", "constrained_rule",
           "constrained_value", "exceptional_law"]

_EPS = 1e-12


def blip(Q1, Q0):
    r""":math:`B(W) = \bar Q(1,W) - \bar Q(0,W)`."""
    a = [float(v) for v in k.vec(Q1)]
    b = [float(v) for v in k.vec(Q0)]
    if len(a) != len(b):
        raise ValueError("tloilr: the two arms differ in length")
    return [a[i] - b[i] for i in range(len(a))]


def resource_threshold(B, kappa):
    r"""The :math:`(1-\kappa)` quantile of the blip, floored at zero.

    Never negative: treating a unit with a negative blip wastes budget
    and harms the unit, so the threshold is
    :math:`\max(0, q_{1-\kappa})`, and equals zero exactly when the
    constraint does not bind.
    """
    b = sorted(float(v) for v in k.vec(B))
    kp = float(kappa)
    if not 0.0 < kp <= 1.0:
        raise ValueError("tloilr: kappa must lie in (0,1], got %r"
                         % (kappa,))
    n = len(b)
    idx = int(math.ceil((1.0 - kp) * n)) - 1
    idx = min(max(idx, 0), n - 1)
    q = b[idx]
    binding = sum(1 for v in b if v > 0.0) > kp * n
    return {"tau": max(0.0, q), "quantile": q, "kappa": kp,
            "binding": binding,
            "fraction_positive_blip": sum(1 for v in b
                                          if v > 0.0) / float(n),
            "note": "tau = 0 exactly when the budget is not binding, "
                    "which recovers the unconstrained rule"}


def constrained_rule(B, kappa):
    r""":math:`d_\kappa(W) = I\{B(W) > \tau_\kappa\}`."""
    b = [float(v) for v in k.vec(B)]
    t = resource_threshold(b, kappa)
    d = [1.0 if v > t["tau"] else 0.0 for v in b]
    return {"rule": d, "tau": t["tau"],
            "treated_fraction": sum(d) / len(d),
            "binding": t["binding"]}


def constrained_value(Q1, Q0, kappa):
    r""":math:`E[Y_{d_\kappa}]` and the cost of the constraint."""
    q1 = [float(v) for v in k.vec(Q1)]
    q0 = [float(v) for v in k.vec(Q0)]
    B = blip(q1, q0)
    r = constrained_rule(B, kappa)
    n = len(q1)
    val = sum(q1[i] if r["rule"][i] == 1.0 else q0[i]
              for i in range(n)) / n
    unc = sum(max(q1[i], q0[i]) for i in range(n)) / n
    return RichResult(payload={
        "estimate": val, "value": val,
        "unconstrained_value": unc, "cost_of_constraint": unc - val,
        "tau": r["tau"], "treated_fraction": r["treated_fraction"],
        "kappa": float(kappa), "binding": r["binding"],
        "method": "optimal resource-constrained value; van der Laan & "
                  "Rose (2018) Chap. 23",
        "note": "a binding constraint makes the estimation problem "
                "EASIER: the condition concerns the blip's density at "
                "tau > 0 rather than the absence of an atom at zero",
    })


def exceptional_law(B, tol=1e-9):
    r"""How much blip mass sits exactly at zero.

    An atom at zero makes the *unconstrained* optimal rule non-unique
    and its value non-pathwise-differentiable -- the exceptional law.
    An active constraint moves the relevant point away from zero and
    the problem becomes regular again.
    """
    b = [float(v) for v in k.vec(B)]
    n = len(b)
    at_zero = sum(1 for v in b if abs(v) <= float(tol))
    return {"mass_at_zero": at_zero / float(n),
            "exceptional": at_zero > 0,
            "n_at_zero": at_zero,
            "note": "exceptional laws break regular estimation of the "
                    "UNCONSTRAINED optimal value; the constrained "
                    "problem is unaffected when tau > 0"}


def cheatsheet():
    return ("tloilr: at most a proportion kappa can be treated, so the "
            "rule is a THRESHOLD on the blip B(W) = Q(1,W) - Q(0,W): "
            "treat the largest blips until the budget runs out, "
            "tau = max(0, (1-kappa) quantile), and tau = 0 recovers "
            "the unconstrained rule. The constraint makes inference "
            "EASIER: regular estimation of the unconstrained value "
            "needs a NONEXCEPTIONAL law (no atom of blip at zero), "
            "while an ACTIVE constraint with continuous covariates "
            "only needs a condition at tau > 0 -- far more reasonable, "
            "and root-n estimation follows.")


# compact alias per ledger/NAMING.md
resourceconstrainedrule = constrained_value
