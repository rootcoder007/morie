# morie.fn -- function file (rootcoder007/morie)
r"""Bounds in a short dynamic discrete-choice panel.

The model is the dynamic random-effects binary choice

.. math:: y_{it} = 1\{x_{it}'\beta + y_{i,t-1}\gamma + \alpha_i
          + \varepsilon_{it} \ge 0\},

and the awkward part is not :math:`\alpha_i` but :math:`y_{i1}`. A
random-effects likelihood needs :math:`f_1(y_{i1} \mid x_i, \alpha_i)`,
and that depends on how the covariates evolved *before* the sample
began -- the initial conditions problem. A fixed-effects treatment
sidesteps it but is only known for special cases and typically rules
out time effects.

**What this module does instead: assume nothing about either.** Leave
both :math:`G(\alpha \mid x)` and the initial-condition distribution
completely unrestricted, and ask which :math:`(\beta, \gamma)` remain
*consistent with the observed choice sequences*. A short panel has
:math:`2^T` possible sequences, so the data are a probability vector
over those cells. For a candidate :math:`\theta`, each value of
:math:`\alpha` implies a sequence-probability vector, and any mixing
distribution over :math:`\alpha` gives a convex combination of them.
So :math:`\theta` is in the identified set exactly when the observed
vector lies in the **convex hull** of the model's sequence
probabilities.

That is a linear feasibility problem, which is why the paper describes
its calculations as simple and constructive: no assumption about
:math:`G` is needed, because :math:`G` is exactly the mixing weights
being solved for.

**Two findings, and they point in opposite directions.** Point
identification usually **fails** -- the feasible set has positive
width, so no amount of data pins :math:`\theta` down. But the feasible
set is often **very small**, so the failure may not matter in
practice. Both are properties of the returned set rather than claims
about it: ``identified_set`` reports the width, and the anchor
measures that it is positive and that it shrinks as :math:`T` grows.

**Conditioning on the first observation is not a fix.** Working with
the likelihood conditional on :math:`y_{i1}` gives convenient
functional forms but is not internally consistent across different
panel lengths. This module therefore treats :math:`y_{i1}` as a free
initial state and lets the mixing distribution place mass on
:math:`(\alpha, y_1)` jointly.

References
----------
Honoré, B. E. & Tamer, E. (2006) "Bounds on Parameters in Panel
Dynamic Discrete Choice Models", *Econometrica* 74(3), 611-629,
doi:10.1111/j.1468-0262.2006.00676.x. Sec. 1 (the initial conditions
problem and why neither the random-effects nor the fixed-effects route
avoids it), eq. (2) (the model implemented here), and Sec. 2.1
(identification: what is learned about theta without assumptions on
G(alpha | x) or f_1(y_1 | x, alpha), and the constructive calculation
of the identified region).

Honoré, B. E. & Kyriazidou, E. (2000) "Panel Data Discrete Choice
Models with Lagged Dependent Variables", *Econometrica* 68(4),
839-874, doi:10.1111/1468-0262.00139. The fixed-effects estimator
requiring matched covariates that Sec. 1 contrasts this approach with.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["sequence_probabilities", "in_identified_set",
           "identified_set", "sequence_frequencies"]

_EPS = 1e-9


def _logit(z):
    return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, z))))


def sequence_probabilities(beta, gamma, x, alpha, y0, link="logit"):
    r"""Probability of each :math:`2^T` choice sequence given
    :math:`(\theta, \alpha, y_0)`.

    ``x`` is a length-T list of covariate rows for one individual.
    ``y0`` is the state entering the first modelled period, which is
    left free rather than modelled -- that is the initial conditions
    problem, sidestepped by treating it as part of the mixing
    distribution.
    """
    xs = [[float(v) for v in r] for r in k.mat(x)]
    T = len(xs)
    if T < 1:
        raise ValueError("bnshrt: need at least one period")
    b = [float(v) for v in k.vec(beta)]
    if len(b) != len(xs[0]):
        raise ValueError("bnshrt: beta has %d entries for %d "
                         "covariates" % (len(b), len(xs[0])))
    g = float(gamma)
    a = float(alpha)
    if link not in ("logit", "probit"):
        raise ValueError("bnshrt: link must be logit or probit, got "
                         "%r" % (link,))
    F = _logit if link == "logit" else (lambda z: k.pnorm(z))
    out = {}
    for code in range(2 ** T):
        seq = [(code >> t) & 1 for t in range(T)]
        p, prev = 1.0, int(y0)
        for t in range(T):
            idx = sum(xs[t][j] * b[j] for j in range(len(b))) \
                + g * prev + a
            pt = F(idx)
            p *= pt if seq[t] == 1 else (1.0 - pt)
            prev = seq[t]
        out[tuple(seq)] = p
    return out


def sequence_frequencies(Y):
    r"""Observed frequencies of each choice sequence."""
    rows = [tuple(int(v) for v in r) for r in k.mat(Y)]
    n = len(rows)
    if n == 0:
        raise ValueError("bnshrt: no observations")
    T = len(rows[0])
    if any(len(r) != T for r in rows):
        raise ValueError("bnshrt: all sequences must have the same "
                         "length")
    if any(v not in (0, 1) for r in rows for v in r):
        raise ValueError("bnshrt: choices must be 0/1")
    counts = {}
    for r in rows:
        counts[r] = counts.get(r, 0) + 1
    return {seq: counts.get(seq, 0) / float(n)
            for seq in [tuple((c >> t) & 1 for t in range(T))
                        for c in range(2 ** T)]}


def in_identified_set(freq, beta, gamma, x, alpha_grid,
                      y0_values=(0, 1), link="logit", tol=1e-4,
                      iters=4000):
    r"""Is :math:`\theta` consistent with the observed frequencies?

    Solves the feasibility problem: do there exist non-negative mixing
    weights over :math:`(\alpha, y_0)`, summing to one, whose implied
    sequence probabilities reproduce ``freq``? Solved by projected
    gradient on the squared discrepancy, which is convex in the
    weights.

    Returns the attained discrepancy; a value at or below ``tol`` means
    :math:`\theta` cannot be ruled out.
    """
    cols = []
    for a in alpha_grid:
        for y0 in y0_values:
            sp = sequence_probabilities(beta, gamma, x, a, y0,
                                        link=link)
            cols.append(sp)
    if not cols:
        raise ValueError("bnshrt: the alpha grid is empty")
    keys = sorted(freq)
    A = [[c[kk] for c in cols] for kk in keys]
    target = [float(freq[kk]) for kk in keys]
    m = len(cols)
    R = len(keys)
    w = [1.0 / m] * m
    # Step size from the actual curvature. A fixed step is wrong here:
    # the columns are probability vectors whose scale depends on T and
    # on the alpha grid, so the Lipschitz constant of the gradient
    # (the largest eigenvalue of 2 A'A) varies by orders of magnitude
    # between problems. Estimate it by power iteration and use 1/L.
    v = [1.0] * m
    L = 1.0
    for _ in range(60):
        Av = [sum(A[r][j] * v[j] for j in range(m)) for r in range(R)]
        AtAv = [2.0 * sum(A[r][j] * Av[r] for r in range(R))
                for j in range(m)]
        nrm = math.sqrt(sum(x * x for x in AtAv))
        if nrm <= _EPS:
            break
        v = [x / nrm for x in AtAv]
        L = nrm
    step = 1.0 / max(L, _EPS)
    # accelerated projected gradient (FISTA), which converges on this
    # convex problem rather than oscillating
    y_acc, t_acc, prev = list(w), 1.0, list(w)
    for _ in range(int(iters)):
        pred = [sum(A[r][j] * y_acc[j] for j in range(m))
                for r in range(R)]
        grad = [2.0 * sum((pred[r] - target[r]) * A[r][j]
                          for r in range(R)) for j in range(m)]
        w = _project_simplex([y_acc[j] - step * grad[j]
                              for j in range(m)])
        t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t_acc * t_acc))
        mom = (t_acc - 1.0) / t_new
        y_acc = [w[j] + mom * (w[j] - prev[j]) for j in range(m)]
        prev, t_acc = list(w), t_new
    pred = [sum(A[r][j] * w[j] for j in range(m))
            for r in range(len(keys))]
    disc = math.sqrt(sum((pred[r] - target[r]) ** 2
                         for r in range(len(keys))))
    return {"discrepancy": disc, "feasible": disc <= float(tol),
            "weights": w, "fitted": pred, "target": target}


def _project_simplex(v):
    """Euclidean projection onto the probability simplex."""
    n = len(v)
    u = sorted(v, reverse=True)
    css = 0.0
    rho, theta = 0, 0.0
    for i in range(n):
        css += u[i]
        t = (css - 1.0) / (i + 1)
        if u[i] - t > 0:
            rho, theta = i + 1, t
    return [max(x - theta, 0.0) for x in v]


def identified_set(Y, x, beta_grid, gamma_grid, alpha_grid,
                   beta_fixed=None, link="logit", tol=1e-3):
    r"""The set of :math:`(\beta_1, \gamma)` not ruled out by the data.

    Sweeps a grid and keeps the feasible points. The width of the
    result is the paper's headline: usually positive -- so point
    identification fails -- but often small.
    """
    freq = sequence_frequencies(Y)
    keep, disc = [], {}
    for bv in beta_grid:
        for gv in gamma_grid:
            b = [bv] if beta_fixed is None else [bv] + list(beta_fixed)
            r = in_identified_set(freq, b, gv, x, alpha_grid,
                                  link=link, tol=tol)
            disc[(bv, gv)] = r["discrepancy"]
            if r["feasible"]:
                keep.append((bv, gv))
    if not keep:
        return RichResult(payload={
            "estimate": None, "set": [], "n_feasible": 0,
            "discrepancy": disc,
            "note": "no grid point is feasible at this tolerance -- "
                    "either the grid misses the identified set or the "
                    "model is rejected",
        })
    bs = [p[0] for p in keep]
    gs = [p[1] for p in keep]
    return RichResult(payload={
        "estimate": (sum(bs) / len(bs), sum(gs) / len(gs)),
        "set": keep, "n_feasible": len(keep),
        "beta_bounds": (min(bs), max(bs)),
        "gamma_bounds": (min(gs), max(gs)),
        "beta_width": max(bs) - min(bs),
        "gamma_width": max(gs) - min(gs),
        "point_identified": (max(bs) - min(bs) < _EPS
                             and max(gs) - min(gs) < _EPS),
        "discrepancy": disc,
        "method": "identified set by mixture feasibility over "
                  "(alpha, y0); Honore & Tamer (2006) Sec. 2.1",
        "assumes": "nothing about G(alpha | x) or the initial "
                   "condition distribution",
    })


def cheatsheet():
    return ("bnshrt: short dynamic panel probit/logit. The initial "
            "conditions problem means f_1(y_1|x,alpha) is unknown, so "
            "leave BOTH it and G(alpha|x) unrestricted. theta is in "
            "the identified set iff the observed sequence frequencies "
            "lie in the CONVEX HULL of the model's sequence "
            "probabilities over (alpha, y_0) -- a linear feasibility "
            "problem. Point identification usually FAILS (positive "
            "width) but the set is often small.")


# compact alias per ledger/NAMING.md
shortpanelbound = identified_set
