# morie.fn -- function file (rootcoder007/morie)
r"""Optimal individualized treatment under a resource constraint.

Luedtke & van der Laan (2016). At most a proportion :math:`\kappa` of
the population can be treated, and the question is who. The paper's
Sec. 2 answers it in closed form.

Write the blip function -- the conditional treatment effect given
whatever summary :math:`V` the rule is allowed to depend on --

.. math:: \bar Q_{b,P}(v) = E_P\!\left[\bar Q_P(1, W)
                            - \bar Q_P(0, W) \mid V = v\right],

let :math:`S_P(\tau) = \Pr_P(\bar Q_{b,P} > \tau)` be its survival
function, and set

.. math:: \eta_P = \inf\{\tau : S_P(\tau) \le \kappa\},
          \qquad \tau_P = \max\{\eta_P,\, 0\}.

The optimal resource-constrained deterministic rule is then

.. math:: \tilde d_P(v) = I\!\left(\bar Q_{b,P}(v) > \tau_P\right),

which their Theorem 1 shows is optimal subject to
:math:`E_P[\tilde d(V)] \le \kappa`, provided
:math:`\Pr_P(\bar Q_{b,P}(V) = \tau_P) = 0`.

**The max with zero is not a detail.** If the constraint is slack --
fewer people have a positive blip than :math:`\kappa` allows -- then
:math:`\eta_P` is negative and clipping it at zero stops the rule
treating people the treatment would harm merely because there is
capacity spare. Drop the max and the rule fills its quota regardless of
sign, which is both wrong and the kind of thing that looks fine on a
value comparison because the harmed patients are few. The anchor sets
:math:`\kappa` deliberately larger than the treatable fraction and
checks nobody with a negative blip is treated.

**Deterministic and stochastic rules are different problems.** With a
tie at the threshold the deterministic problem is a 0-1 knapsack, which
the paper notes is NP-hard; the stochastic version relaxes it to the
fractional knapsack, which is easy and can hit the budget exactly. Both
are here: ``rule="deterministic"`` is Theorem 1, ``rule="stochastic"``
randomises among the boundary stratum to spend the remaining budget.

References
----------
Luedtke, A. R. & van der Laan, M. J. (2016) "Optimal individualized
treatments in resource-limited settings", *The International Journal of
Biostatistics* 12(1), 283-303, doi:10.1515/ijb-2015-0007. Sec. 2, eq.
(1)-(3) and Theorem 1.

van der Laan, M. J. & Rose, S. (eds.) (2018) *Targeted Learning in
Data Science*, Springer Series in Statistics,
doi:10.1007/978-3-319-65304-4, Ch. 12 "Optimal Individualized
Treatments Under Limited Resources".
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["threshold_treatment_msm", "blip_function", "rc_threshold",
           "rc_rule", "rule_value"]

_RULES = ("deterministic", "stochastic")


def blip_function(y, A, W, V=None, ridge=1e-8):
    r"""Qbar_b(v) = E[Qbar(1,W) - Qbar(0,W) | V = v].

    The outcome model is fitted with a treatment-covariate interaction,
    so the blip is allowed to vary; without the interaction it would be
    a constant and every rule would tie.
    """
    yv, av = k.vec(y), k.vec(A)
    n = len(yv)
    if len(av) != n:
        raise ValueError("blip_function: %d outcomes but %d treatments"
                         % (n, len(av)))
    if any(v not in (0.0, 1.0) for v in av):
        raise ValueError("blip_function: treatment must be binary 0/1")
    Wm = k.mat(W) if W is not None else [[] for _ in range(n)]
    p = len(Wm[0]) if Wm and Wm[0] else 0
    Z = k.design([[av[i]] + list(Wm[i])
                  + [av[i] * Wm[i][j] for j in range(p)]
                  for i in range(n)], n)
    b = k.lstsq(Z, yv, ridge)

    def q_at(a, i):
        row = ([1.0, a] + list(Wm[i])
               + [a * Wm[i][j] for j in range(p)])
        return sum(b[t] * row[t] for t in range(len(b)))

    blip_w = [q_at(1.0, i) - q_at(0.0, i) for i in range(n)]
    if V is None:
        return blip_w, {"coef": b, "q1": [q_at(1.0, i) for i in range(n)],
                        "q0": [q_at(0.0, i) for i in range(n)]}
    # E[blip | V]: regress the blip on the summary V
    Vm = k.mat(V)
    Zv = k.design(Vm, n)
    bv = k.lstsq(Zv, blip_w, ridge)
    proj = k.matvec(Zv, bv)
    return list(proj), {"coef": b, "v_coef": bv,
                        "blip_w": blip_w,
                        "q1": [q_at(1.0, i) for i in range(n)],
                        "q0": [q_at(0.0, i) for i in range(n)]}


def rc_threshold(blip, kappa):
    r"""tau = max{eta, 0} with eta = inf{tau : S(tau) <= kappa}.

    S is the empirical survival function of the blip, so eta is the
    (1 - kappa) quantile: the smallest cut at which no more than a
    kappa fraction lies strictly above.
    """
    kap = float(kappa)
    if not 0.0 < kap < 1.0:
        raise ValueError("rc_threshold: kappa must be in (0,1), got %r"
                         % (kappa,))
    b = sorted(float(v) for v in blip)
    n = len(b)
    if n == 0:
        raise ValueError("rc_threshold: empty blip")

    def surv(t):
        return sum(1 for v in b if v > t) / float(n)

    # eta is attained at one of the observed values (or just below the
    # smallest), so searching them is exact rather than a grid guess.
    eta = b[-1]
    for v in [b[0] - 1.0] + b:
        if surv(v) <= kap:
            eta = v
            break
    tau = max(eta, 0.0)
    return tau, {"eta": eta, "survival_at_tau": surv(tau),
                 "constraint_active": eta > 0.0}


def rc_rule(blip, kappa, rule="deterministic", seed=0):
    """The optimal resource-constrained rule, Theorem 1.

    Returns the per-subject treatment probability. The deterministic
    rule gives 0/1; the stochastic rule randomises within the boundary
    stratum so the budget can be met exactly -- the fractional knapsack
    relaxation the paper points to.
    """
    if rule not in _RULES:
        raise ValueError("rc_rule: rule must be 'deterministic' or "
                         "'stochastic', got %r" % (rule,))
    b = [float(v) for v in blip]
    n = len(b)
    tau, info = rc_threshold(b, kappa)
    d = [1.0 if v > tau else 0.0 for v in b]
    if rule == "deterministic":
        return d, dict(info, tau=tau, treated_fraction=sum(d) / n)

    # spend whatever budget the strict inequality left, on the boundary
    used = sum(d) / n
    spare = float(kappa) - used
    at_tau = [i for i in range(n) if b[i] == tau]
    if spare > 1e-12 and at_tau and tau > 0.0:
        share = min(1.0, spare * n / len(at_tau))
        for i in at_tau:
            d[i] = share
    return d, dict(info, tau=tau, treated_fraction=sum(d) / n,
                   boundary_share=(d[at_tau[0]] if at_tau else 0.0))


def rule_value(q1, q0, d):
    """Psi_d = E[Qbar(d(V), W)], the value of a rule."""
    n = len(q1)
    return sum(d[i] * q1[i] + (1.0 - d[i]) * q0[i]
               for i in range(n)) / n


def threshold_treatment_msm(y, A, W, threshold_grid=None, kappa=0.1,
                            V=None, rule="deterministic", seed=0):
    r"""Optimal resource-constrained rule and its value.

    Parameters
    ----------
    y, A, W : array-like
        Outcome, binary treatment, covariates.
    threshold_grid : array-like, optional
        Extra thresholds at which to report the value, so the trade-off
        between budget and value is visible rather than asserted.
    kappa : float
        The resource constraint: at most this proportion may be treated.
    V : array-like, optional
        The summary the rule may depend on. Defaults to all of W.
    rule : {"deterministic", "stochastic"}

    Returns
    -------
    RichResult
        ``estimate`` is the value of the optimal rule under the
        constraint; ``tau`` the threshold, ``rule`` the per-subject
        treatment probabilities.

    Examples
    --------
    A budget smaller than the treatable fraction binds::

        r = threshold_treatment_msm(y, A, W, kappa=0.2)
        r["tau"], r["treated_fraction"]
    """
    blip, info = blip_function(y, A, W, V=V)
    d, rinfo = rc_rule(blip, kappa, rule=rule, seed=seed)
    q1, q0 = info["q1"], info["q0"]
    n = len(q1)

    curve = []
    if threshold_grid is not None:
        for t in [float(v) for v in k.vec(threshold_grid)]:
            dt = [1.0 if blip[i] > t else 0.0 for i in range(n)]
            curve.append({"tau": t, "treated_fraction": sum(dt) / n,
                          "value": rule_value(q1, q0, dt)})

    treat_all = rule_value(q1, q0, [1.0] * n)
    treat_none = rule_value(q1, q0, [0.0] * n)
    unconstrained = [1.0 if v > 0.0 else 0.0 for v in blip]

    return RichResult(payload={
        "estimate": rule_value(q1, q0, d),
        "value": rule_value(q1, q0, d),
        "tau": rinfo["tau"],
        "eta": rinfo["eta"],
        "constraint_active": rinfo["constraint_active"],
        "rule": d,
        "treated_fraction": rinfo["treated_fraction"],
        "blip": blip,
        "value_treat_all": treat_all,
        "value_treat_none": treat_none,
        "value_unconstrained": rule_value(q1, q0, unconstrained),
        "unconstrained_fraction": sum(unconstrained) / n,
        "threshold_curve": curve,
        "kappa": float(kappa), "n": n, "rule_kind": rule,
        "method": "optimal resource-constrained ITR, Luedtke & van der "
                  "Laan (2016) Sec. 2 Theorem 1",
    })


def cheatsheet():
    return ("thrtmt: optimal treatment under a budget kappa. blip "
            "Qb(v)=E[Q(1,W)-Q(0,W)|V=v]; eta=inf{t: P(Qb>t)<=kappa}; "
            "tau=max(eta,0); rule d(v)=I(Qb(v)>tau) (Luedtke-vdL 2016 "
            "Thm 1). The max with 0 stops spare capacity treating "
            "people the treatment harms.")


# compact alias per ledger/NAMING.md
thresholdtreatmentmsm = threshold_treatment_msm
