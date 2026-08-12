r"""Proximal gradient and FISTA for composite problems.

Beck, A., & Teboulle, M. (2009) "A Fast Iterative Shrinkage-Thresholding
Algorithm for Linear Inverse Problems", *SIAM J. Imaging Sciences*
**2**(1), 183-202.

Minimises :math:`F(x) = f(x) + g(x)` with :math:`f` smooth with
:math:`L`-Lipschitz gradient and :math:`g` prox-friendly. ISTA is

.. math:: x_{k} = \mathrm{prox}_{g/L}\big(x_{k-1} - \tfrac1L \nabla f(x_{k-1})\big),

and FISTA (their eq. 4.1-4.3) adds the extrapolation

.. math:: t_{k+1} = \frac{1 + \sqrt{1 + 4t_k^2}}{2}, \qquad
          y_{k+1} = x_k + \frac{t_k - 1}{t_{k+1}}(x_k - x_{k-1}),

which improves the rate from :math:`O(1/k)` to :math:`O(1/k^2)` at the
same cost per iteration. That gap is the whole point of the paper and is
what the anchors check.

For the lasso, :math:`g = \lambda\lVert x\rVert_1` and the prox is
soft-thresholding, :math:`\mathrm{sign}(x)\max(|x| - \tau, 0)`.

Routes
------
``accelerate=True`` is FISTA, ``False`` is plain ISTA; both are the
paper's, and ISTA is kept because it is monotone while FISTA is not.
``backtrack=True`` uses the paper's Sec. 4 backtracking line search on
L, for when the Lipschitz constant is unknown.
"""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["prox_gradient", "soft_threshold", "lasso_fista", "prxgms"]


def soft_threshold(v, tau):
    """Prox of tau * |.|_1, applied elementwise."""
    out = []
    for x in v:
        x = float(x)
        if x > tau:
            out.append(x - tau)
        elif x < -tau:
            out.append(x + tau)
        else:
            out.append(0.0)
    return out


def prox_gradient(fun, grad, prox, x0, L=1.0, max_iter=500, tol=1e-10,
                  accelerate=True, backtrack=False, eta=2.0, g_fun=None):
    r"""Proximal gradient / FISTA on ``f + g``."""
    x = [float(v) for v in np.atleast_1d(np.asarray(x0, dtype=float))]
    n = len(x)
    L = float(L)
    if L <= 0.0:
        raise ValueError("prox_gradient: L must be positive, got %r" % (L,))
    y = list(x)
    t = 1.0
    prev = list(x)
    obj = []
    it = 0
    converged = False

    for it in range(1, int(max_iter) + 1):
        gy = [float(v) for v in grad(y)]
        Lk = L
        if backtrack:
            fy = float(fun(y))
            for _ in range(60):
                z = prox([y[j] - gy[j] / Lk for j in range(n)], 1.0 / Lk)
                d = [z[j] - y[j] for j in range(n)]
                q = (fy + sum(gy[j] * d[j] for j in range(n))
                     + 0.5 * Lk * sum(v * v for v in d))
                if float(fun(z)) <= q + 1e-15:
                    break
                Lk *= eta
            L = Lk
        else:
            z = prox([y[j] - gy[j] / Lk for j in range(n)], 1.0 / Lk)

        if accelerate:
            t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            w = (t - 1.0) / t_next
            y = [z[j] + w * (z[j] - prev[j]) for j in range(n)]
            t = t_next
        else:
            y = z

        step = math.sqrt(sum((z[j] - prev[j]) ** 2 for j in range(n)))
        prev = list(z)
        fz = float(fun(z))
        obj.append(fz + (float(g_fun(z)) if g_fun is not None else 0.0))
        if step <= tol:
            converged = True
            break

    return RichResult(payload={
        "estimate": prev,
        "x": prev,
        "fun": float(fun(prev)),
        "objective": obj,
        "iterations": int(it),
        "L": float(L),
        "accelerated": bool(accelerate),
        "converged": bool(converged),
        "method": ("FISTA (Beck & Teboulle 2009, eq. 4.1-4.3)" if accelerate
                   else "ISTA (Beck & Teboulle 2009, Sec. 2)"),
    })


def lasso_fista(A, b, lam, max_iter=500, tol=1e-10, accelerate=True):
    r"""Lasso: minimise 1/2 ||Ax - b||^2 + lam ||x||_1."""
    Am = [[float(v) for v in row] for row in A]
    bv = [float(v) for v in b]
    n_rows = len(Am)
    p = len(Am[0]) if n_rows else 0
    lam = float(lam)

    def f(x):
        r = [sum(Am[i][j] * x[j] for j in range(p)) - bv[i]
             for i in range(n_rows)]
        return 0.5 * sum(v * v for v in r)

    def g(x):
        r = [sum(Am[i][j] * x[j] for j in range(p)) - bv[i]
             for i in range(n_rows)]
        return [sum(Am[i][j] * r[i] for i in range(n_rows)) for j in range(p)]

    # L = largest eigenvalue of A'A, by power iteration.
    v = [1.0] * p
    L = 1.0
    for _ in range(200):
        w = g([0.0] * p)
        Av = [sum(Am[i][j] * v[j] for j in range(p)) for i in range(n_rows)]
        u = [sum(Am[i][j] * Av[i] for i in range(n_rows)) for j in range(p)]
        nrm = math.sqrt(sum(t * t for t in u))
        if nrm <= 0.0:
            break
        v = [t / nrm for t in u]
        L = nrm
    L = max(L, 1e-12)

    # The prox of lam*|.|_1 at step 1/L thresholds at lam/L, NOT at
    # 1/L. Passing bare soft_threshold drops lam out of the iteration
    # entirely and silently solves a different problem -- the objective
    # still *reports* the lam term, so it looks plausible while being
    # wrong for every lam != 1.
    def _prox(v, t):
        return soft_threshold(v, lam * t)

    res = prox_gradient(f, g, _prox, [0.0] * p, L=L,
                        max_iter=max_iter, tol=tol, accelerate=accelerate,
                        g_fun=lambda x: lam * sum(abs(t) for t in x))
    d = dict(res)
    d["lambda"] = lam
    d["L"] = L
    return RichResult(payload=d)


def cheatsheet():
    return ("prxgms: ISTA/FISTA, x = prox_{g/L}(x - grad f / L), "
            "t_{k+1} = (1+sqrt(1+4t^2))/2, y = x + (t-1)/t_next (x - x_prev); "
            "soft threshold for the lasso.")


prxgms = prox_gradient
