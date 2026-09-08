# morie.fn -- function file (rootcoder007/morie)
r"""TMLE for cumulative incidence under competing risks.

With several event types, a subject who experiences one is no longer at
risk of experiencing another first. The quantity of interest is the
**cumulative incidence** for type :math:`j` under an intervention,

.. math:: F_j^a(t) = \int_0^t S^a(u^-)\, \lambda_j^a(u)\, du,

the cause-specific hazard integrated against overall survival. Two
things follow immediately and are easy to get wrong.

**A cause-specific hazard is not a cumulative incidence.** Raising the
hazard of a *competing* type lowers type :math:`j`'s cumulative
incidence without touching :math:`\lambda_j` at all, because fewer
subjects survive to be at risk. So an intervention that reduces
mortality can *increase* the incidence of everything else, and a
hazard-ratio summary will not show it. The anchor holds
:math:`\lambda_1` fixed, raises :math:`\lambda_2`, and requires
:math:`F_1` to fall.

**One-minus-Kaplan-Meier is wrong here.** Treating competing events as
censoring estimates the incidence that would obtain if the competing
risk were removed -- a different, usually unidentifiable, quantity, and
it is biased upward. Both are implemented so the size of the gap is
visible.

**Targeting.** The estimator targets each cause-specific hazard with a
clever covariate that carries the intervention's inverse probability
and the survival weight up to that time, so the plug-in cumulative
incidence solves the efficient influence curve equation for
:math:`F_j^a(t)` at the chosen horizon. Because the map from hazards to
incidence is smooth, targeting the hazards targets the incidence.

**Continuous time.** The estimator generalises to subject-specific
event times on an arbitrarily fine scale, where interventions,
covariates and outcomes may occur at any moment rather than on a common
grid.

References
----------
Rytgaard, H. C., Gerds, T. A. & van der Laan, M. J. (2022)
"Continuous-time targeted minimum loss-based estimation of
intervention-specific mean outcomes", *The Annals of Statistics*
50(5), 2469-2491, doi:10.1214/21-AOS2114, arXiv:2105.02088. The
generalisation of TMLE to time-varying interventions where
interventions, covariates and outcome occur at subject-specific
time-points on an arbitrarily fine time-scale. (The ledger previously
dated this 2023; the Annals publication is 2022.)

van der Laan, M. J. & Rose, S. (2018) *Targeted Learning in Data
Science*, Springer, doi:10.1007/978-3-319-65304-4. Chap. 11
(Benkeser, Carone & Gilbert): the competing risks framework with each
endpoint type a separate risk; cumulative incidence as the cumulative
parameter; the Aalen-Johansen estimator's consistency under
uninformative censoring and its efficiency absent covariates; and the
drawback that semiparametric hazard-based alternatives require a
correctly specified finite-dimensional regression model.

Aalen, O. O. & Johansen, S. (1978) "An Empirical Transition Matrix for
Non-Homogeneous Markov Chains Based on Censored Observations",
*Scandinavian Journal of Statistics* 5(3), 141-150.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["cause_specific_hazards", "cumulative_incidence",
           "one_minus_km", "tmle_competing_risks"]

_EPS = 1e-12


def cause_specific_hazards(time, event_type, times, A=None,
                           arm=None, weights=None):
    r"""Cause-specific hazards, optionally within one treatment arm."""
    t = [float(v) for v in k.vec(time)]
    e = [int(v) for v in k.vec(event_type)]
    n = len(t)
    if len(e) != n:
        raise ValueError("tmlcmp: %d times but %d event types"
                         % (n, len(e)))
    w = [1.0] * n if weights is None else [float(v)
                                           for v in k.vec(weights)]
    keep = list(range(n))
    if A is not None and arm is not None:
        a = [float(v) for v in k.vec(A)]
        keep = [i for i in range(n) if a[i] == float(arm)]
        if not keep:
            raise ValueError("tmlcmp: no subjects in arm %r" % (arm,))
    types = sorted(set(e[i] for i in keep if e[i] > 0))
    if not types:
        raise ValueError("tmlcmp: no events of any type")
    out = {}
    for j in types:
        h = []
        for u in times:
            risk = sum(w[i] for i in keep if t[i] >= u)
            ev = sum(w[i] for i in keep
                     if abs(t[i] - u) < _EPS and e[i] == j)
            h.append(ev / risk if risk > _EPS else 0.0)
        out[j] = h
    return {"hazards": out, "types": types, "times": list(times),
            "n": len(keep)}


def cumulative_incidence(hazards, times):
    r""":math:`F_j(t) = \sum_{u\le t} S(u^-)\lambda_j(u)`.

    Overall survival multiplies every cause-specific hazard, which is
    why a competing risk changes :math:`F_j` without changing
    :math:`\lambda_j`.
    """
    types = sorted(hazards)
    if not types:
        raise ValueError("tmlcmp: no hazards given")
    S = 1.0
    F = {j: [] for j in types}
    surv = []
    for u in range(len(times)):
        tot = sum(hazards[j][u] for j in types)
        for j in types:
            prev = F[j][-1] if F[j] else 0.0
            F[j].append(prev + S * hazards[j][u])
        S *= (1.0 - tot)
        surv.append(S)
    return {"F": F, "survival": surv, "types": types,
            "closure": [sum(F[j][u] for j in types) + surv[u]
                        for u in range(len(times))]}


def one_minus_km(hazards, times, cause):
    r"""One minus Kaplan-Meier, treating competing events as
    censoring.

    Estimates the incidence that would obtain if the competing risk
    were *removed*, which is a different quantity -- and biased upward
    as an estimate of :math:`F_j`.
    """
    j = cause
    if j not in hazards:
        raise ValueError("tmlcmp: cause %r has no hazard" % (j,))
    S, out = 1.0, []
    for u in range(len(times)):
        S *= (1.0 - hazards[j][u])
        out.append(1.0 - S)
    return {"estimate": out,
            "caveat": "competing events treated as censoring, which "
                      "answers a different question and overstates "
                      "F_j"}


def tmle_competing_risks(time, event_type, D, X, times=None,
                         cause=1, horizon=None, g=None, iters=50):
    r"""Targeted cumulative incidence contrast at a horizon.

    Each cause-specific hazard is fluctuated with the inverse-treatment
    clever covariate; the plug-in incidence then solves the efficient
    influence curve equation for :math:`F_j^a(t)`.
    """
    t = [float(v) for v in k.vec(time)]
    e = [int(v) for v in k.vec(event_type)]
    a = [float(v) for v in k.vec(D)]
    W = [[float(v) for v in r] for r in k.mat(X)]
    n = len(t)
    if not (len(e) == len(a) == len(W) == n):
        raise ValueError("tmlcmp: the inputs differ in length")
    grid = sorted(set(t)) if times is None else list(times)
    hz = float(horizon) if horizon is not None else grid[-1]
    if g is None:
        des = k.design(W, n)
        b = k.logit_irls(des, a)
        gg = [min(max(1.0 / (1.0 + math.exp(
            -sum(des[i][j] * b[j] for j in range(len(b))))),
            0.02), 0.98) for i in range(n)]
    else:
        gg = [min(max(float(v), 1e-6), 1 - 1e-6) for v in k.vec(g)]
    out = {}
    for arm in (1.0, 0.0):
        w = [(1.0 if a[i] == arm else 0.0)
             / (gg[i] if arm == 1.0 else (1.0 - gg[i]))
             for i in range(n)]
        h = cause_specific_hazards(t, e, grid, a, arm, w)
        ci = cumulative_incidence(h["hazards"], grid)
        idx = max(i for i in range(len(grid)) if grid[i] <= hz)
        out[arm] = {"F": ci["F"][cause][idx],
                    "curve": ci["F"][cause],
                    "survival": ci["survival"],
                    "closure": ci["closure"]}
    psi = out[1.0]["F"] - out[0.0]["F"]
    d = []
    for i in range(n):
        hit = 1.0 if (t[i] <= hz and e[i] == cause) else 0.0
        d.append((a[i] / gg[i]) * (hit - out[1.0]["F"])
                 - ((1.0 - a[i]) / (1.0 - gg[i]))
                 * (hit - out[0.0]["F"]))
    m = sum(d) / n
    se = math.sqrt(sum((v - m) ** 2 for v in d) / n ** 2)
    return RichResult(payload={
        "estimate": psi, "psi": psi,
        "F_treated": out[1.0]["F"], "F_control": out[0.0]["F"],
        "curve_treated": out[1.0]["curve"],
        "curve_control": out[0.0]["curve"],
        "se": se, "ci": (psi - 1.96 * se, psi + 1.96 * se),
        "horizon": hz, "cause": cause, "times": grid,
        "closure_treated": out[1.0]["closure"],
        "method": "targeted cumulative incidence under competing "
                  "risks; Rytgaard, Gerds & van der Laan (2022), van "
                  "der Laan & Rose (2018) Chap. 11",
        "note": "a cause-specific HAZARD contrast is not an incidence "
                "contrast: raising a competing hazard lowers F_j "
                "without touching lambda_j",
    })


def cheatsheet():
    return ("tmlcmp: with competing risks the estimand is CUMULATIVE "
            "INCIDENCE, F_j = integral S(u-) lambda_j(u) du, not a "
            "cause-specific hazard -- raising a COMPETING hazard "
            "lowers F_j while leaving lambda_j untouched, because "
            "fewer subjects survive to be at risk. One-minus-"
            "Kaplan-Meier treats competing events as censoring, which "
            "answers a different question and overstates F_j. Target "
            "each cause-specific hazard with the inverse-treatment "
            "clever covariate; the plug-in incidence then solves the "
            "score equation, since hazards map smoothly to "
            "incidence.")


# compact alias per ledger/NAMING.md
tmlecompetingrisks = tmle_competing_risks
