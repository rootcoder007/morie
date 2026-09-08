# morie.fn -- function file (rootcoder007/morie)
"""Derivative of a smoothed function."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["smoothed_derivative", "derivative_function"]


def smoothed_derivative(x, y, at=None, order=1, n_basis=None, lam=1e-4,
                        penalty_order=None):
    r"""Derivative from a penalised spline, differentiated analytically.

    A global polynomial (monomial) basis is fitted with a roughness
    penalty on the
    :math:`m`-th derivative,

    .. math::
       \min_c \; \lVert y - \Phi c\rVert^2 + \lambda\, c'Rc,
       \qquad R_{jk} = \int \phi_j^{(m)}(t)\,\phi_k^{(m)}(t)\,dt,

    and the derivative is obtained by differentiating the fitted
    basis rather than by differencing the fitted values.

    That distinction is the whole reason this exists. Finite
    differencing an estimate amplifies noise by :math:`1/h`, so a
    curve that looks smooth can have a derivative that is pure noise --
    and each further derivative multiplies the problem. Ramsay and
    Silverman's point is that the SMOOTHING must be chosen for the
    derivative you intend to use, not for the fit: a penalty that
    yields a good-looking curve routinely yields an unusable second
    derivative.

    The rule of thumb they give is that penalising order :math:`m`
    leaves derivatives up to :math:`m-2` well behaved, so estimating a
    second derivative wants a fourth-order penalty. That is applied
    automatically when ``penalty_order`` is not set, and
    ``penalty_adequate`` reports whether it holds.

    ``noise_amplification`` compares the derivative's roughness against
    the fit's, which is the diagnostic that tells you the smoothing was
    too light before you interpret anything.

    Parameters
    ----------
    x, y : array-like, shape (n,)
    at : array-like, optional
        Evaluation points; the input grid by default.
    order : int
        Derivative order to return.
    n_basis : int, optional
    lam : float
        Roughness penalty.
    penalty_order : int, optional
        Derivative penalised. ``order + 2`` by default.

    Returns
    -------
    RichResult
        ``derivative``, ``fitted``, ``at``, ``effective_df``,
        ``noise_amplification``, ``penalty_adequate``,
        ``finite_difference`` (for contrast).

    References
    ----------
    Ramsay and Silverman (2005), *Functional Data Analysis*, 2nd ed.,
    Springer, chapters 3-5.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 1, 200)
    >>> out = smoothed_derivative(t, t ** 2, order=1)
    >>> bool(abs(float(np.mean(out["derivative"])) - 1.0) < 0.1)
    True
    """
    xv = np.asarray(x, dtype=float).ravel()
    yv = np.asarray(y, dtype=float).ravel()
    n = xv.size
    if yv.size != n:
        raise ValueError("x and y must agree in length.")
    if n < 6:
        raise ValueError("need at least 6 points, got %d." % n)
    order = int(order)
    if order < 0:
        raise ValueError("order must be non-negative.")
    m = order + 2 if penalty_order is None else int(penalty_order)
    K = int(n_basis) if n_basis else min(max(n // 4, m + 2), 40)
    if K < m + 2:
        raise ValueError(
            "n_basis must exceed the penalty order by at least 2."
        )
    o = np.argsort(xv)
    xs, ys = xv[o], yv[o]
    lo, hi = float(xs[0]), float(xs[-1])
    if hi <= lo:
        raise ValueError("x must span a positive range.")

    # scaled monomial basis, differentiated exactly; a polynomial basis
    # keeps the derivative and the penalty both closed-form
    def design(t, d=0):
        u = (np.asarray(t, dtype=float) - lo) / (hi - lo)
        cols = []
        for j in range(K):
            if d == 0:
                cols.append(u ** j)
            elif j < d:
                cols.append(np.zeros_like(u))
            else:
                c = 1.0
                for q in range(d):
                    c *= (j - q)
                cols.append(c * u ** (j - d))
        return np.column_stack(cols)

    Phi = design(xs)
    # exact integral of the product of m-th derivatives over [0, 1]
    R = np.zeros((K, K))
    for j in range(K):
        for k in range(K):
            if j < m or k < m:
                continue
            cj = ck = 1.0
            for q in range(m):
                cj *= (j - q)
                ck *= (k - q)
            p = (j - m) + (k - m)
            R[j, k] = cj * ck / (p + 1.0)
    scale = float(np.mean(np.diag(Phi.T @ Phi))) or 1.0
    A = Phi.T @ Phi + lam * scale * R + 1e-10 * np.eye(K)
    c = np.linalg.solve(A, Phi.T @ ys)

    grid = xs if at is None else np.asarray(at, dtype=float).ravel()
    fitted = design(grid) @ c
    deriv = design(grid, order) @ c / (hi - lo) ** order

    H = Phi @ np.linalg.solve(A, Phi.T)
    edf = float(np.trace(H))
    fd = np.gradient(ys, xs) if order == 1 else None
    rough_f = float(np.mean(np.diff(fitted) ** 2)) if fitted.size > 1 else 0.0
    rough_d = float(np.mean(np.diff(deriv) ** 2)) if deriv.size > 1 else 0.0
    return RichResult(
        payload={
            "estimate": deriv,
            "derivative": deriv,
            "fitted": fitted,
            "at": grid,
            "order": order,
            "penalty_order": m,
            "penalty_adequate": bool(m >= order + 2),
            "penalty_note": (
                "penalising order m leaves derivatives up to m-2 usable, so "
                "a second derivative wants a fourth-order penalty; the "
                "smoothing must be chosen for the derivative you intend to "
                "use, not for the fit"
            ),
            "effective_df": edf,
            "lambda": float(lam),
            "n_basis": K,
            "finite_difference": fd,
            "noise_amplification": (float(rough_d / rough_f)
                                    if rough_f > 0 else np.inf),
            "amplification_note": (
                "roughness of the derivative against the fit's; finite "
                "differencing amplifies noise by 1/h and each further "
                "derivative multiplies it, so a smooth-looking curve can "
                "have a derivative that is pure noise"
            ),
            "n": int(n),
            "method": "Order-%d derivative from a penalised smooth" % order,
        }
    )


def cheatsheet():
    return (
        "derivf: derivative by differentiating a penalised basis, with the "
        "penalty-order rule and a noise-amplification check"
    )


#: Catalogue alias for :func:`smoothed_derivative`.
derivative_function = smoothed_derivative
