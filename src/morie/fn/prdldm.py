# morie.fn -- function file (rootcoder007/morie)
"""Proximal forward-backward splitting (the proximal gradient method).

Source FETCHED and read: Combettes, P. L. & Pesquet, J.-C. (2011),
"Proximal splitting methods in signal processing", arXiv:0912.3522,
retrieved as https://ar5iv.labs.arxiv.org/html/0912.3522 .  This is the
survey by the author of Combettes & Wajs (2005), *Multiscale Modeling
and Simulation* 4(4):1168-1200, which the ledger cites and which is
paywalled.  Algorithm 3.4 of the survey, "Constant-step forward-backward
algorithm", equation (21), reads

    y_n     = x_n - beta^-1 grad f_2(x_n)
    lambda_n in [eps, 3/2 - eps]
    x_{n+1} = x_n + lambda_n ( prox_{beta^-1 f_1} y_n - x_n )

with eps in ]0, 3/4[.  Taking lambda_n = 1 and beta^-1 = lr collapses
this to the familiar proximal gradient recursion

    x_{n+1} = prox_{lr g}( x_n - lr grad f(x_n) )

which is what this module implements, with the relaxation left as a
parameter so the full Algorithm 3.4 is reachable.  Proposition 3.5 of
the survey is the convergence statement.

``prox_g`` is supplied by the caller and receives the step size, i.e.
it computes prox_{lr g}; that is the survey's prox_{beta^-1 f_1}.  The
iteration runs a FIXED number of steps, since an early exit on one
language arm and not the other would silently break Python/R parity.
"""

import math

from ._richresult import RichResult

__all__ = ["prox_method"]


def prox_method(f, grad_f, prox_g, x0, lr, n_iter=200, relaxation=1.0):
    """Minimise f + g by forward-backward splitting.

    Parameters
    ----------
    f : callable
        The smooth part, taking a list of floats and returning a float.
        Used only to report the objective; the iteration needs grad_f.
    grad_f : callable
        Gradient of f, returning a list of floats.
    prox_g : callable
        ``prox_g(y, lr)`` returns the proximity operator of lr * g at y.
    x0 : sequence of float
        Starting point.
    lr : float
        Step size, the survey's beta^-1.  Convergence needs lr < 2 / L
        with L the Lipschitz constant of grad f.
    n_iter : int
        Fixed number of iterations.
    relaxation : float
        The survey's lambda_n, held constant here.  Must lie in
        ]0, 3/2[; 1.0 gives the plain proximal gradient step.

    Returns
    -------
    RichResult
        Keys ``x``, ``objective``, ``n_iter``, ``lr``, ``relaxation``,
        ``step_norm``, ``method``.
    """
    x = [float(v) for v in x0]
    lr = float(lr)
    relaxation = float(relaxation)
    if lr <= 0.0:
        raise ValueError("lr must be positive")
    if not (0.0 < relaxation < 1.5):
        raise ValueError("relaxation must lie strictly inside ]0, 3/2[")
    step_norm = 0.0
    for _n in range(int(n_iter)):
        g = [float(v) for v in grad_f(x)]
        if len(g) != len(x):
            raise ValueError("grad_f returned the wrong length")
        y = [x[k] - lr * g[k] for k in range(len(x))]
        z = [float(v) for v in prox_g(y, lr)]
        if len(z) != len(x):
            raise ValueError("prox_g returned the wrong length")
        nx = [x[k] + relaxation * (z[k] - x[k]) for k in range(len(x))]
        step_norm = math.sqrt(sum((nx[k] - x[k]) ** 2 for k in range(len(x))))
        x = nx
    return RichResult(
        payload={
            "x": x,
            "objective": float(f(x)),
            "n_iter": int(n_iter),
            "lr": lr,
            "relaxation": relaxation,
            "step_norm": step_norm,
            "method": "forward-backward splitting, Combettes & Pesquet "
                      "(2011) Algorithm 3.4",
        }
    )


def cheatsheet():
    return "prdldm: Proximal gradient method"


# compact alias per ledger/NAMING.md
proxmethod = prox_method
