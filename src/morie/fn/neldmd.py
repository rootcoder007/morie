"""Nelder-Mead simplex minimisation.

Nelder, J. A. and Mead, R. (1965). A simplex method for function minimization. Computer
Journal 7, 308-313; coefficients and ordering rules as in Lagarias, Reeds, Wright and
Wright (1998), SIAM Journal on Optimization 9, 112-147.
"""

from ._richresult import RichResult

__all__ = ["nelder_mead"]


def nelder_mead(f, x0, step=None, xtol=1e-10, ftol=1e-10, max_iter=None):
    r"""Minimise f from x0 with the Nelder-Mead simplex (reflect 1, expand 2, contract 1/2, shrink 1/2).

    The initial simplex is x0 plus step_i e_i with step_i = 0.1 max(1, |x0_i|) unless
    given. Each iteration sorts the vertices (ties keep the older vertex first),
    reflects the worst through the centroid of the rest, and expands, contracts
    outside or inside, or shrinks toward the best vertex following Lagarias et al.
    (1998, Sec. 2). Stops when max_i |x_i - x_best| <= xtol and
    max_i |f_i - f_best| <= ftol, or after max_iter iterations (default 200 n).

    Parameters
    ----------
    f : callable
    x0 : sequence of float
    step : float or sequence, optional
    xtol, ftol : float
    max_iter : int, optional

    Returns
    -------
    RichResult
        Keys: x, fun, n_iter, n_fev, converged, simplex.

    References
    ----------
    Nelder, J. A. and Mead, R. (1965). Computer Journal 7, 308-313.
    Lagarias, J. C., Reeds, J. A., Wright, M. H. and Wright, P. E. (1998). SIAM J. Optim. 9, 112-147.

    Examples
    --------
    >>> r = nelder_mead(lambda x: (x[0] - 1) ** 2 + (x[1] + 2) ** 2, [0.0, 0.0])
    >>> [round(v, 6) for v in r["x"]]
    [1.0, -2.0]
    """
    x0 = [float(v) for v in x0]
    n = len(x0)
    if n == 0:
        raise ValueError("x0 must be non-empty")
    if step is None:
        st = [0.1 * max(1.0, abs(v)) for v in x0]
    elif isinstance(step, (int, float)):
        st = [float(step)] * n
    else:
        st = [float(v) for v in step]
    max_iter = 200 * n if max_iter is None else int(max_iter)
    simp = [list(x0)] + [[x0[j] + (st[i] if j == i else 0.0) for j in range(n)] for i in range(n)]
    fv = [float(f(v)) for v in simp]
    nfev, it, conv = n + 1, 0, False
    while it < max_iter:
        order = sorted(range(n + 1), key=lambda i: (fv[i], i))
        simp = [simp[i] for i in order]
        fv = [fv[i] for i in order]
        if (
            max(abs(simp[i][j] - simp[0][j]) for i in range(1, n + 1) for j in range(n)) <= xtol
            and max(abs(v - fv[0]) for v in fv) <= ftol
        ):
            conv = True
            break
        it += 1
        cen = [sum(simp[i][j] for i in range(n)) / n for j in range(n)]
        xr = [2 * c - w for c, w in zip(cen, simp[n])]
        fr = float(f(xr))
        nfev += 1
        if fr < fv[0]:
            xe = [3 * c - 2 * w for c, w in zip(cen, simp[n])]
            fe = float(f(xe))
            nfev += 1
            simp[n], fv[n] = (xe, fe) if fe < fr else (xr, fr)
            continue
        if fr < fv[n - 1]:
            simp[n], fv[n] = xr, fr
            continue
        if fr < fv[n]:
            xc = [c + 0.5 * (r - c) for c, r in zip(cen, xr)]
            fc = float(f(xc))
            nfev += 1
            if fc <= fr:
                simp[n], fv[n] = xc, fc
                continue
        else:
            xc = [c + 0.5 * (w - c) for c, w in zip(cen, simp[n])]
            fc = float(f(xc))
            nfev += 1
            if fc < fv[n]:
                simp[n], fv[n] = xc, fc
                continue
        for i in range(1, n + 1):
            simp[i] = [b + 0.5 * (v - b) for b, v in zip(simp[0], simp[i])]
            fv[i] = float(f(simp[i]))
        nfev += n
    order = sorted(range(n + 1), key=lambda i: (fv[i], i))
    simp = [simp[i] for i in order]
    fv = [fv[i] for i in order]
    return RichResult(
        title="Nelder-Mead minimisation",
        summary_lines=[("f", fv[0]), ("iterations", it)],
        payload={"x": simp[0], "fun": fv[0], "n_iter": it, "n_fev": nfev, "converged": conv, "simplex": simp},
    )


def cheatsheet():
    return "neldmd: Nelder-Mead simplex minimiser (Lagarias et al. 1998 rules)"
