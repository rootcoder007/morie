r"""Frank-Wolfe (conditional gradient) for quadratic programs.

Frank, M., & Wolfe, P. (1956) "An algorithm for quadratic programming",
*Naval Research Logistics Quarterly* **3**(1-2), 95-110; Wolfe, P.
(1959) "The simplex method for quadratic programming", *Econometrica*
**27**(3), 382-398.

Minimises a convex :math:`f` over a compact convex set :math:`C` using
only *linear* minimisation over :math:`C` -- never a projection:

.. math:: s^k = \arg\min_{s \in C} \langle \nabla f(x^k), s\rangle,
          \qquad x^{k+1} = (1-\gamma_k)x^k + \gamma_k s^k .

Because each iterate is a convex combination of vertices, feasibility is
maintained exactly and for free; that is the reason to use it on the
simplex, where projection is comparatively awkward.

The **Frank-Wolfe gap**

.. math:: g_k = \langle \nabla f(x^k), x^k - s^k\rangle \;\ge\;
          f(x^k) - f(x^\star)

is a certificate computed at no extra cost, since :math:`s^k` is
already known. It is returned as ``gap``: unlike a gradient norm it
bounds the true suboptimality, so a caller can stop on it and know what
they have.

Routes
------
``step`` selects the step size:

``"exact"``
    Exact line search on the quadratic, :math:`\gamma = \mathrm{clip}
    (g_k / (d'Qd), 0, 1)` with :math:`d = s - x`. Available because the
    objective is quadratic, and much faster than the default schedule.
``"standard"``
    :math:`\gamma_k = 2/(k+2)`, the classical schedule, which needs no
    knowledge of :math:`Q` and gives the :math:`O(1/k)` rate.

``domain`` selects :math:`C`: ``"simplex"`` (the probability simplex,
whose linear minimiser is the unit vector at the smallest gradient
coordinate) or ``"box"`` with bounds, whose minimiser takes each
coordinate to whichever end the gradient prefers.
"""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["frank_wolfe_qp", "qpdual"]

_STEPS = ("exact", "standard")
_DOMAINS = ("simplex", "box")


def _lmo(gradient, domain, lower, upper):
    """Linear minimisation oracle: argmin_{s in C} <grad, s>."""
    n = len(gradient)
    if domain == "simplex":
        # The minimum of a linear function over the simplex is attained
        # at a vertex: all mass on the smallest gradient coordinate.
        j = 0
        for i in range(1, n):
            if gradient[i] < gradient[j]:
                j = i
        s = [0.0] * n
        s[j] = 1.0
        return s
    return [(lower[i] if gradient[i] > 0.0 else upper[i]) for i in range(n)]


def frank_wolfe_qp(Q, c, x0=None, domain="simplex", lower=None, upper=None,
                   step="exact", max_iter=1000, tol=1e-12):
    r"""Minimise :math:`\tfrac12 x'Qx + c'x` over the simplex or a box.

    Returns
    -------
    RichResult
        ``gap`` is the Frank-Wolfe gap at the returned point, an upper
        bound on :math:`f(x) - f(x^\star)`.
    """
    Qm = [[float(v) for v in row] for row in Q]
    n = len(Qm)
    if any(len(r) != n for r in Qm):
        raise ValueError("frank_wolfe_qp: Q must be square")
    cv = [float(v) for v in np.atleast_1d(np.asarray(c, dtype=float))]
    if len(cv) != n:
        raise ValueError(
            "frank_wolfe_qp: c has length %d but Q is %dx%d"
            % (len(cv), n, n))
    dom = str(domain).lower()
    if dom not in _DOMAINS:
        raise ValueError(
            "frank_wolfe_qp: domain must be one of %s, got %r"
            % (", ".join(_DOMAINS), domain))
    st = str(step).lower()
    if st not in _STEPS:
        raise ValueError(
            "frank_wolfe_qp: step must be one of %s, got %r"
            % (", ".join(_STEPS), step))

    if dom == "box":
        if lower is None or upper is None:
            raise ValueError("frank_wolfe_qp: domain='box' needs lower and upper")
        lo = [float(v) for v in np.atleast_1d(np.asarray(lower, dtype=float))]
        hi = [float(v) for v in np.atleast_1d(np.asarray(upper, dtype=float))]
        for i in range(n):
            if lo[i] > hi[i]:
                raise ValueError(
                    "frank_wolfe_qp: lower[%d] exceeds upper[%d]" % (i, i))
        x = [0.5 * (lo[i] + hi[i]) for i in range(n)] if x0 is None else \
            [float(v) for v in np.atleast_1d(np.asarray(x0, dtype=float))]
    else:
        lo = hi = None
        x = [1.0 / n] * n if x0 is None else \
            [float(v) for v in np.atleast_1d(np.asarray(x0, dtype=float))]

    def grad(v):
        return [sum(Qm[i][j] * v[j] for j in range(n)) + cv[i]
                for i in range(n)]

    def obj(v):
        return (0.5 * sum(v[i] * Qm[i][j] * v[j]
                          for i in range(n) for j in range(n))
                + sum(cv[i] * v[i] for i in range(n)))

    gap = float("inf")
    it = 0
    converged = False
    history = [obj(x)]
    for it in range(int(max_iter)):
        g = grad(x)
        s = _lmo(g, dom, lo, hi)
        d = [s[i] - x[i] for i in range(n)]
        gap = -sum(g[i] * d[i] for i in range(n))
        if gap <= tol:
            converged = True
            break
        if st == "exact":
            dQd = sum(d[i] * Qm[i][j] * d[j]
                      for i in range(n) for j in range(n))
            gamma = 1.0 if dQd <= 0.0 else min(1.0, max(0.0, gap / dQd))
        else:
            gamma = 2.0 / (it + 2.0)
        x = [x[i] + gamma * d[i] for i in range(n)]
        history.append(obj(x))

    return RichResult(payload={
        "estimate": x,
        "x": x,
        "fun": float(obj(x)),
        "gap": float(gap),
        # it is 0-based, so the number of iterations performed
        # is it + 1
        "iterations": int(it) + 1,
        "converged": bool(converged),
        "domain": dom,
        "step": st,
        "history": history,
        "method": "Frank-Wolfe conditional gradient "
                  "(Frank & Wolfe 1956); gap bounds f(x) - f(x*)",
    })


def cheatsheet():
    return ("qpdual: Frank-Wolfe, s = argmin_C <grad, s>, x += gamma (s - x); "
            "gap = <grad, x - s> >= f(x) - f*; steps exact / 2/(k+2); "
            "domains simplex / box.")


qpdual = frank_wolfe_qp

# public names resolved by fn/_lazy_map.json
quadratic_program = frank_wolfe_qp
