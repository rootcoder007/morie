"""BFGS quasi-Newton minimisation with a strong-Wolfe line search.

Nocedal, J. and Wright, S. J. (2006). Numerical Optimization, 2nd ed., Algorithm 6.1.
"""

from ._qncore import dot, num_grad, wolfe
from ._richresult import RichResult

__all__ = ["bfgs_minimize"]


def bfgs_minimize(f, x0, grad=None, gtol=1e-8, max_iter=1000):
    r"""Minimise f from x0 by BFGS (inverse-Hessian form).

    H_{k+1} = (I - rho s y') H_k (I - rho y s') + rho s s', rho = 1/(y's),
    x_{k+1} = x_k - a_k H_k g_k with a_k satisfying the strong Wolfe conditions
    (c1 = 1e-4, c2 = 0.9); H_0 = I, rescaled by y's/y'y before the first update
    (N&W eq. 6.20). Stops when max |g_i| <= gtol.

    Parameters
    ----------
    f : callable
        Objective on a list of floats.
    x0 : sequence of float
    grad : callable, optional
        Gradient; central differences when omitted.
    gtol : float
    max_iter : int

    Returns
    -------
    RichResult
        Keys: x, fun, grad, n_iter, n_fev, converged.

    References
    ----------
    Nocedal, J. and Wright, S. J. (2006). Numerical Optimization, Algorithm 6.1.

    Examples
    --------
    >>> r = bfgs_minimize(lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2, [-1.2, 1.0])
    >>> [round(v, 8) for v in r["x"]]
    [1.0, 1.0]
    """
    gr = grad if grad is not None else (lambda x: num_grad(f, x))
    x = [float(v) for v in x0]
    n = len(x)
    fx, g = float(f(x)), [float(v) for v in gr(x)]
    H = [[float(i == j) for j in range(n)] for i in range(n)]
    nfev, it, conv = 1, 0, max(abs(v) for v in g) <= gtol
    while not conv and it < max_iter:
        p = [-dot(H[i], g) for i in range(n)]
        if dot(p, g) >= 0:  # lost descent: restart from steepest descent
            H = [[float(i == j) for j in range(n)] for i in range(n)]
            p = [-v for v in g]
        cache = {}

        def phi(a, p=p, x=x, cache=cache):
            xa = [xi + a * pi for xi, pi in zip(x, p)]
            fa, ga = float(f(xa)), [float(v) for v in gr(xa)]
            cache[a] = (xa, fa, ga)
            return fa, dot(ga, p)

        a, _, _, ne = wolfe(phi)
        nfev += ne
        if a == 0.0:
            break
        xn, fn, gn = cache[a]
        s = [u - v for u, v in zip(xn, x)]
        y = [u - v for u, v in zip(gn, g)]
        sy = dot(s, y)
        it += 1
        if sy > 1e-12 * max(1.0, dot(y, y)):
            if it == 1:
                sc = sy / dot(y, y)
                H = [[sc * H[i][j] for j in range(n)] for i in range(n)]
            rho = 1.0 / sy
            Hy = [dot(H[i], y) for i in range(n)]
            yHy = dot(y, Hy)
            H = [
                [
                    H[i][j] - rho * (Hy[i] * s[j] + s[i] * Hy[j]) + (rho * rho * yHy + rho) * s[i] * s[j]
                    for j in range(n)
                ]
                for i in range(n)
            ]
        stalled = abs(fx - fn) <= 1e-15 * max(1.0, abs(fx)) and max(abs(v) for v in s) <= 1e-15 * max(
            1.0, max(abs(v) for v in xn)
        )
        x, fx, g = xn, fn, gn
        conv = max(abs(v) for v in g) <= gtol
        if stalled:
            break
    return RichResult(
        title="BFGS minimisation",
        summary_lines=[("f", fx), ("iterations", it)],
        payload={"x": x, "fun": fx, "grad": g, "n_iter": it, "n_fev": nfev, "converged": conv},
    )


def cheatsheet():
    return "bfgsmin: BFGS quasi-Newton minimiser with strong-Wolfe line search (Nocedal and Wright Alg 6.1)"
