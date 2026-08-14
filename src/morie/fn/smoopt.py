# morie.fn -- function file (rootcoder007/morie)
r"""Platt's SMO: the same dual, chosen differently.

:mod:`svmopt` solves the SVM dual by taking the **maximal violating
pair** at every step. Platt's original SMO answers the same question
-- which two multipliers to move -- with a pair of **heuristics**, and
the difference is worth keeping rather than collapsing: the maximal
pair needs the whole gradient, while Platt's needs only the errors of
the non-bound examples, which is what let SMO run on problems that did
not fit in memory in 1998.

**Why two variables, again.** The linear equality constraint
:math:`\sum y_i\alpha_i = 0` means the smallest change that keeps
feasibility moves **two** multipliers -- and at two the QP has an
analytic solution, so SMO invokes **no inner QP solver at all**. That
is the entire reason the method is small enough to state in a page.

**The outer loop alternates deliberately.** One pass over *all*
examples, then repeated passes over the **non-bound** examples
(:math:`0 < \alpha_i < C`) until none of them violates KKT, then back
to a full pass. Bound examples tend to stay bound, so sweeping them
repeatedly is wasted work; but they cannot be ignored forever, or a
violator sitting at a bound is never noticed. ``outer_loop_schedule``
encodes exactly that alternation.

**The inner heuristic maximises the step it can take, not the
violation.** Given a first index with error :math:`E_1`, choose the
second to maximise :math:`|E_1 - E_2|`, because the analytic step
length is proportional to that difference -- a cheap proxy for
progress. When it makes no progress Platt falls back: try the
non-bound examples from a random start, then all examples, and only
then abandon the first index. ``second_choice`` implements the
hierarchy, and returns which level fired.

**Mind the sign convention.** Platt writes the decision function as
:math:`f(x) = \sum_j \alpha_j y_j K(x_j, x) - b`, with the threshold
**subtracted**; LIBSVM and :mod:`svmopt` add it. The two solvers
therefore report thresholds of opposite sign for the same separator --
here :math:`b = 1` against :math:`b = -1`, both placing the boundary
at the same point. Comparing the raw numbers across the two modules
without accounting for that would look like a bug and is not one.

**The threshold is recomputed, not accumulated.** After each step
:math:`b` is derived from the multipliers that ended up non-bound, and
when both are at a bound any value between the two candidates
satisfies KKT -- so the midpoint is taken. Accumulating :math:`b`
incrementally would let rounding error drift into the decision
function.

References
----------
Platt, J. C. (1998) "Sequential Minimal Optimization: A Fast
Algorithm for Training Support Vector Machines", Microsoft Research
Technical Report MSR-TR-98-14. [PDF supplied by Vee.] The
decomposition to the smallest possible optimisation problem -- two
Lagrange multipliers, because the linear equality constraint forces
them to move together -- solved analytically so that no numerical QP
optimisation is required; the outer loop alternating between single
passes over the entire training set and repeated passes over the
non-bound examples until all of them obey the KKT conditions within
tolerance; the second-choice heuristic maximising |E_1 - E_2| as a
proxy for the step size, with the fallback hierarchy over non-bound
examples from a random position and then over all examples; and the
computation of the threshold b from the resulting non-bound
multipliers, taking the midpoint when both are at a bound.

Chang, C.-C. & Lin, C.-J. (2011) "LIBSVM: A Library for Support
Vector Machines", *ACM TIST* 2(3), Article 27,
doi:10.1145/1961189.1961199. The maximal-violating-pair selection
kept as the alternative route; implemented in :mod:`svmopt`.

Cortes, C. & Vapnik, V. (1995) "Support-Vector Networks", *Machine
Learning* 20(3), 273-297, doi:10.1007/BF00994018. The dual being
solved.
"""

import math

from . import _array_core as np
from . import _s03core as k
from . import svmopt as sq
from ._richresult import RichResult

__all__ = ["error_cache", "violates_kkt", "outer_loop_schedule",
           "second_choice", "compute_threshold", "smo_platt"]

_EPS = 1e-12


def error_cache(alpha, y, K, b):
    r""":math:`E_i = f(x_i) - y_i` for every example."""
    a = [float(v) for v in k.vec(alpha)]
    yy = [float(v) for v in k.vec(y)]
    n = len(a)
    out = []
    for i in range(n):
        f = sum(a[j] * yy[j] * K[i][j] for j in range(n)) - float(b)
        out.append(f - yy[i])
    return out


def violates_kkt(i, alpha, y, E, C, tol=1e-3):
    r"""KKT for one example, within tolerance.

    :math:`y_iE_i < -\tau` with :math:`\alpha_i < C`, or
    :math:`y_iE_i > \tau` with :math:`\alpha_i > 0`.
    """
    a = float(alpha[i])
    r = float(y[i]) * float(E[i])
    return (r < -float(tol) and a < float(C) - _EPS) or \
        (r > float(tol) and a > _EPS)


def outer_loop_schedule(alpha, C, examine_all):
    r"""Which examples this pass looks at.

    A full pass, then non-bound passes until they are all clean, then
    a full pass again. Bound examples rarely move, but skipping them
    forever would hide a violator sitting at a bound.
    """
    a = [float(v) for v in k.vec(alpha)]
    if examine_all:
        return {"indices": list(range(len(a))), "kind": "all",
                "note": "a full sweep catches violators at a bound"}
    nb = [i for i in range(len(a)) if _EPS < a[i] < float(C) - _EPS]
    return {"indices": nb, "kind": "non_bound",
            "n_non_bound": len(nb),
            "note": "the non-bound set is where the action is"}


def second_choice(i1, alpha, y, E, C, rng, tol=1e-3):
    r"""Platt's hierarchy for the second index.

    1. maximise :math:`|E_1 - E_2|` over the non-bound examples;
    2. failing progress, the non-bound examples from a random start;
    3. then all examples from a random start.
    """
    a = [float(v) for v in k.vec(alpha)]
    n = len(a)
    nb = [i for i in range(n) if _EPS < a[i] < float(C) - _EPS
          and i != int(i1)]
    if len(nb) > 1:
        j = max(nb, key=lambda t: abs(E[int(i1)] - E[t]))
        return {"index": j, "level": 1,
                "gap": abs(E[int(i1)] - E[j]),
                "note": "the analytic step is proportional to "
                        "|E1 - E2|, so this maximises progress"}
    start = int(float(rng.uniform()) * max(n, 1)) % max(n, 1)
    if nb:
        j = nb[start % len(nb)]
        return {"index": j, "level": 2,
                "note": "non-bound examples from a random position"}
    for t in range(n):
        j = (start + t) % n
        if j != int(i1):
            return {"index": j, "level": 3,
                    "note": "all examples from a random position"}
    return {"index": None, "level": 4,
            "note": "no second index available; abandon this i1"}


def compute_threshold(i1, i2, a1_new, a2_new, alpha, y, E, K, b, C):
    r"""Recompute :math:`b` from whichever multiplier is non-bound.

    Both at a bound leaves an interval of valid thresholds, and the
    midpoint is the standard choice.
    """
    yy = [float(v) for v in k.vec(y)]
    a = [float(v) for v in k.vec(alpha)]
    i, j = int(i1), int(i2)
    d1 = float(a1_new) - a[i]
    d2 = float(a2_new) - a[j]
    b1 = (float(b) + E[i] + yy[i] * d1 * K[i][i]
          + yy[j] * d2 * K[i][j])
    b2 = (float(b) + E[j] + yy[i] * d1 * K[i][j]
          + yy[j] * d2 * K[j][j])
    free1 = _EPS < float(a1_new) < float(C) - _EPS
    free2 = _EPS < float(a2_new) < float(C) - _EPS
    if free1:
        return {"b": b1, "from": "i1", "b1": b1, "b2": b2}
    if free2:
        return {"b": b2, "from": "i2", "b1": b1, "b2": b2}
    return {"b": 0.5 * (b1 + b2), "from": "midpoint",
            "b1": b1, "b2": b2,
            "note": "both at a bound, so any value between b1 and b2 "
                    "satisfies KKT"}


def smo_platt(y, K, C=1.0, tol=1e-3, eps=1e-5, max_passes=200,
              seed=0):
    r"""SMO with Platt's own loops.

    Returns the multipliers, the threshold and the loop statistics --
    the last so the schedule can be inspected rather than assumed.
    """
    yy = [float(v) for v in k.vec(y)]
    n = len(yy)
    if any(v not in (-1.0, 1.0) for v in yy):
        raise ValueError("smoopt: labels must be -1 or +1")
    if float(C) <= 0.0:
        raise ValueError("smoopt: C must be positive")
    rng = np.random.default_rng(seed)
    a = [0.0] * n
    b = 0.0
    examine_all, passes, changed_total = True, 0, 0
    full_passes, nb_passes = 0, 0
    while passes < int(max_passes):
        passes += 1
        E = error_cache(a, yy, K, b)
        sched = outer_loop_schedule(a, C, examine_all)
        if sched["kind"] == "all":
            full_passes += 1
        else:
            nb_passes += 1
        changed = 0
        for i1 in sched["indices"]:
            if not violates_kkt(i1, a, yy, E, C, tol):
                continue
            pick = second_choice(i1, a, yy, E, C, rng, tol)
            i2 = pick["index"]
            if i2 is None:
                continue
            L, H = sq._bounds(i1, i2, a, yy, C)
            if H <= L + _EPS:
                continue
            eta = K[i1][i1] + K[i2][i2] - 2.0 * K[i1][i2]
            if eta <= _EPS:
                continue
            a2_new = a[i2] + yy[i2] * (E[i1] - E[i2]) / eta
            a2_new = min(max(a2_new, L), H)
            if abs(a2_new - a[i2]) < float(eps) * (
                    a2_new + a[i2] + float(eps)):
                continue
            a1_new = a[i1] - yy[i1] * yy[i2] * (a2_new - a[i2])
            th = compute_threshold(i1, i2, a1_new, a2_new, a, yy, E,
                                   K, b, C)
            a[i1], a[i2] = a1_new, a2_new
            b = th["b"]
            E = error_cache(a, yy, K, b)
            changed += 1
        changed_total += changed
        if examine_all:
            examine_all = False
        elif changed == 0:
            examine_all = True
            if passes > 1:
                E = error_cache(a, yy, K, b)
                if not any(violates_kkt(i, a, yy, E, C, tol)
                           for i in range(n)):
                    break
    E = error_cache(a, yy, K, b)
    sv = [i for i in range(n) if a[i] > _EPS]
    return RichResult(payload={
        "estimate": a, "alpha": a, "b": b, "passes": passes,
        "full_passes": full_passes, "non_bound_passes": nb_passes,
        "steps": changed_total, "support_vectors": sv,
        "n_sv": len(sv),
        "equality_residual": sum(a[i] * yy[i] for i in range(n)),
        "kkt_violations": sum(1 for i in range(n)
                              if violates_kkt(i, a, yy, E, C, tol)),
        "objective": sq.dual_objective(a, yy, K),
        "method": "SMO with Platt's heuristics; Platt (1998)",
        "note": "same dual as svmopt, different working-set rule -- "
                "Platt's needs only the non-bound errors; note b "
                "follows Platt's f = sum(a y K) - b, the NEGATIVE of "
                "the LIBSVM convention used in svmopt",
    })


def cheatsheet():
    return ("smoopt: same SVM dual as svmopt, different CHOICE of "
            "pair. Two multipliers because the equality constraint "
            "forces them to move together, and at two the QP is "
            "analytic -- SMO calls NO inner QP solver. Outer loop "
            "ALTERNATES: one full sweep, then repeated sweeps over the "
            "NON-BOUND examples until they all satisfy KKT, then a "
            "full sweep again -- bound examples rarely move, but "
            "skipping them forever hides a violator sitting at a "
            "bound. Inner heuristic maximises |E1 - E2|, since the "
            "analytic step is proportional to it, with a fallback "
            "hierarchy. b is RECOMPUTED each step, not accumulated.")


# compact alias per ledger/NAMING.md
sequential_minimal_optimization = smo_platt
