# morie.fn -- function file (rootcoder007/morie)
r"""One-step TMLE by a universal least favorable submodel.

The ordinary TMLE fluctuates along a **local** least favorable
submodel and iterates: update, recompute the clever covariate,
update again, until the efficient score equation is solved. Each
iteration is a fresh maximum likelihood step, and when the data carry
sparse information about the target, that iteration is where the
estimator becomes unstable.

**The fix is a submodel that is least favorable everywhere, not just
locally.** A submodel :math:`\{P_\epsilon\}` is *universal* least
favorable when its score at **every** :math:`\epsilon` -- not only at
zero -- equals the canonical gradient at the current point:

.. math:: \frac{d}{d\epsilon}\log p_\epsilon = D^*(P_\epsilon).

Then a single move along it, of the length that solves the score
equation, is enough: no iteration, and the path taken is the shortest
one achieving the required bias reduction. The estimator is
:math:`\psi_n^* = \Psi(P_n^1)`, one step.

**Why it is more stable, concretely.** The iterative TMLE recomputes
its direction after each update and can overshoot in the sparse case;
the universal submodel's direction is *defined* by the gradient at
wherever it currently is, so following it is an integral rather than a
sequence of jumps. The construction is a differential equation, solved
here by small steps whose limit is the path -- and ``build_ulfm``
reports the step count so the discretisation is visible rather than
implied.

**It generalises without changing shape.** The same construction
handles a multivariate target parameter, and even an
infinite-dimensional one such as a complete treatment-specific
survival curve, because the submodel is characterised by the canonical
gradient rather than by a finite parametrisation.

References
----------
van der Laan, M. J. & Rose, S. (2018) *Targeted Learning in Data
Science*, Springer, doi:10.1007/978-3-319-65304-4. Chap. 5
(one-dimensional universal least favorable parametric submodels for
univariate, multivariate and infinite-dimensional target parameters;
the definition by which the score at every epsilon equals the
canonical gradient; the resulting one-step TMLE solving the efficient
influence curve equation without iteration; the reading of the
universal submodel as a shortest path achieving the desired bias
reduction; the argument that the iterative TMLE can be unstable when
the data provide sparse information about the target; and the worked
treatment-specific survival example).

van der Laan, M. J. & Gruber, S. (2016) "One-step targeted minimum
loss-based estimation based on universal least favorable
one-dimensional submodels", *International Journal of Biostatistics*
12(1), 351-378, doi:10.1515/ijb-2015-0054. The construction this
chapter relies on.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["build_ulfm", "one_step_tmle", "iterative_tmle",
           "is_universal"]

_EPS = 1e-12


def _logit(p):
    q = min(max(float(p), 1e-9), 1 - 1e-9)
    return math.log(q / (1.0 - q))


def _expit(x):
    return 1.0 / (1.0 + math.exp(-x)) if x > -700 else 0.0


def build_ulfm(Q, H_fn, Y, eps_max=2.0, steps=400):
    r"""Integrate the universal least favorable path.

    The direction is recomputed *continuously* from the current
    point -- ``H_fn(Q_current)`` -- rather than fixed at the start,
    which is what makes the submodel universal rather than local.
    """
    q = [float(v) for v in k.vec(Q)]
    y = [float(v) for v in k.vec(Y)]
    n = len(q)
    if len(y) != n:
        raise ValueError("tl1step: %d fits but %d outcomes"
                         % (n, len(y)))
    if int(steps) < 1:
        raise ValueError("tl1step: steps must be at least 1")
    de = float(eps_max) / int(steps)
    path = [(0.0, list(q))]
    # integrate in BOTH directions: the score at epsilon = 0 may have
    # either sign, and a forward-only path cannot reach the solution
    # when it is negative.
    for sign in (1.0, -1.0):
        cur = list(q)
        for s in range(int(steps)):
            h = [float(v) for v in H_fn(cur)]
            cur = [_expit(_logit(cur[i]) + sign * de * h[i])
                   for i in range(n)]
            path.append((sign * (s + 1) * de, list(cur)))
    path.sort(key=lambda t: t[0])
    return {"path": path, "steps": int(steps), "d_epsilon": de,
            "note": "the direction is recomputed at every point, so "
                    "the submodel is least favorable EVERYWHERE, not "
                    "only at epsilon = 0"}


def is_universal(Q, H_fn, eps=0.3, h=1e-5):
    r"""Check the defining property: score = gradient at every
    :math:`\epsilon`.

    Evaluated away from zero, because at zero a *local* least
    favorable submodel satisfies it too -- so testing only there
    cannot distinguish the two.
    """
    q = [float(v) for v in k.vec(Q)]
    n = len(q)

    def move(e, direction_at_start=False):
        cur = list(q)
        steps = 2000
        de = e / steps
        for _ in range(steps):
            d = H_fn(q) if direction_at_start else H_fn(cur)
            cur = [_expit(_logit(cur[i]) + de * d[i])
                   for i in range(n)]
        return cur

    at = move(eps)
    fwd = move(eps + h)
    num = [(fwd[i] - at[i]) / h for i in range(n)]
    grad = H_fn(at)
    ana = [grad[i] * at[i] * (1.0 - at[i]) for i in range(n)]
    dev = max(abs(num[i] - ana[i]) for i in range(n))
    local = move(eps, True)
    lg = H_fn(local)
    ldev = max(abs(lg[i] - grad[i]) for i in range(n))
    return {"max_deviation": dev, "universal": dev < 1e-3,
            "epsilon": eps,
            "local_submodel_direction_drift": ldev,
            "note": "a LOCAL submodel keeps the direction it had at "
                    "epsilon = 0, so its score no longer equals the "
                    "gradient once it has moved"}


def one_step_tmle(Q, H_fn, Y, eps_max=3.0, steps=600):
    r"""Move once along the universal path to where the score
    vanishes."""
    b = build_ulfm(Q, H_fn, Y, eps_max, steps)
    y = [float(v) for v in k.vec(Y)]
    n = len(y)
    best, chosen = None, b["path"][0]
    for (e, cur) in b["path"]:
        h = H_fn(cur)
        sc = abs(sum(h[i] * (y[i] - cur[i]) for i in range(n)) / n)
        if best is None or sc < best:
            best, chosen = sc, (e, cur)
    e, cur = chosen
    return RichResult(payload={
        "estimate": sum(cur) / n, "psi": sum(cur) / n,
        "epsilon": e, "Q_star": cur, "abs_score": best,
        "iterations": 1, "path_steps": b["steps"],
        "method": "one-step TMLE along a universal least favorable "
                  "submodel; van der Laan & Rose (2018) Chap. 5",
        "note": "no iteration: one move along the shortest path that "
                "achieves the required bias reduction",
    })


def iterative_tmle(Q, H_fn, Y, max_iter=25, tol=1e-8):
    r"""The ordinary iterative TMLE, for comparison.

    The direction is frozen within each iteration and recomputed
    between them -- a sequence of jumps rather than a path.
    """
    q = [float(v) for v in k.vec(Q)]
    y = [float(v) for v in k.vec(Y)]
    n = len(q)
    cur = list(q)
    it = 0
    for it in range(1, int(max_iter) + 1):
        h = [float(v) for v in H_fn(cur)]
        off = [_logit(v) for v in cur]
        e = 0.0
        for _ in range(50):
            p = [_expit(off[i] + e * h[i]) for i in range(n)]
            gr = sum(h[i] * (y[i] - p[i]) for i in range(n))
            he = sum(h[i] * h[i] * p[i] * (1 - p[i])
                     for i in range(n))
            if he < 1e-12:
                break
            e += gr / he
        cur = [_expit(off[i] + e * h[i]) for i in range(n)]
        sc = abs(sum(h[i] * (y[i] - cur[i]) for i in range(n)) / n)
        if sc < float(tol):
            break
    h = H_fn(cur)
    return RichResult(payload={
        "estimate": sum(cur) / n, "psi": sum(cur) / n,
        "iterations": it, "Q_star": cur,
        "abs_score": abs(sum(h[i] * (y[i] - cur[i])
                             for i in range(n)) / n),
        "method": "iterative TMLE along a local least favorable "
                  "submodel",
    })


def cheatsheet():
    return ("tl1step: an ordinary TMLE fluctuates along a LOCAL least "
            "favorable submodel and ITERATES, which is where it "
            "becomes unstable when the data are sparse for the target. "
            "A UNIVERSAL least favorable submodel has score = "
            "canonical gradient at EVERY epsilon, not only at 0, so "
            "one move solves the efficient score equation -- an "
            "integral rather than a sequence of jumps, and the "
            "shortest path achieving the required bias reduction. The "
            "construction is characterised by the gradient, so it "
            "extends to multivariate and infinite-dimensional "
            "targets.")


# compact alias per ledger/NAMING.md
onesteptmle = one_step_tmle
