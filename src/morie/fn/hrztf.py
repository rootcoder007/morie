# morie.fn -- function file (rootcoder007/morie)
"""Fully nonparametric transformation model: both T and F unknown."""

from . import _array_core as np

from ._richresult import RichResult
from .hrztmod import horowitz_transformation_model

__all__ = ["horowitz_both_nonpar_transform"]

_BIG = 1.0e12


def _tn_at(yv, ygrid, T, y2, y1):
    """T_n evaluated at an observation.

    Page 219: T_n(y) "is replaced with an arbitrarily large negative
    number if y < y2 and an arbitrarily large positive number if
    y > y1".  Inside [y2, y1] the grid values are interpolated.
    """
    if yv < y2:
        return -_BIG
    if yv > y1:
        return _BIG
    ny = len(ygrid)
    if yv <= ygrid[0]:
        return T[0]
    for k in range(1, ny):
        if yv <= ygrid[k]:
            lo = ygrid[k - 1]
            hi = ygrid[k]
            if hi <= lo:
                return T[k]
            f = (yv - lo) / (hi - lo)
            return T[k - 1] + f * (T[k] - T[k - 1])
    return T[ny - 1]


def horowitz_both_nonpar_transform(x, y, ny=21, nz=21, nu=25,
                                   bandwidth=None):
    r"""Transformation model with BOTH T and F nonparametric.

    Horowitz (2009), Section 6.3, pages 215-219.  Adds the estimator
    of :math:`F` to the estimator of :math:`T` implemented in
    ``hrztmod``.  Because :math:`U` is independent of :math:`X`,
    :math:`P(U \le u\,|\,a < Z \le b) = F(u)` for any :math:`a, b` in
    the support of :math:`Z`, so for any :math:`y_2 < y_1`

    .. math:: F(u) = P[U \le u \mid T(y_2)-u < Z \le T(y_1)-u]
              = A(u)/B(u)                                     \quad (6.63)

    .. math:: A(u) = E\{I(U \le u)\,
              I[T(y_2)-u < Z \le T(y_1)-u]\}                  \quad (6.64)

    .. math:: B(u) = E\{I[T(y_2)-u < Z \le T(y_1)-u]\}        \quad (6.65)

    and the estimator replaces these by sample analogues (6.66) with
    :math:`U_{ni} = T_n(Y_i) - Z_{ni}`.

    The conditioning is not a technical nicety and the obvious
    shortcut is wrong.  Page 219: "It may seem that F can be estimated
    more simply by the empirical distribution function of
    :math:`U_n = T_n(Y) - X'b_n`, but this is not the case."
    :math:`T` is :math:`n^{-1/2}`-consistent only on a compact
    :math:`[y_2, y_1]` strictly inside the support of :math:`Y`, so
    :math:`U` is effectively observed under CENSORING, and the
    empirical CDF is inconsistent under censoring.  (6.66) is
    consistent despite it.

    Since the indicator set of :math:`A_n` is a subset of that of
    :math:`B_n`, :math:`A_n \le B_n` pointwise and
    :math:`\hat F \in [0, 1]` identically; both are asserted rather
    than assumed.

    Parameters
    ----------
    x : array-like, shape (n,) or (n, d)
        Covariates.
    y : array-like, shape (n,)
        Response.
    ny, nz : int
        Grid sizes passed to the estimator of ``T``.
    nu : int, default 25
        Grid points for ``F``.
    bandwidth : float, optional
        Passed to the estimator of ``T``.

    Returns
    -------
    RichResult
        keys: ``T_hat``, ``F_hat``, ``beta_hat``, ``u_grid``,
        ``A_n``, ``B_n``, ``y_grid``, ``y0``, ``index``,
        ``F_in_unit``, ``A_le_B``, ``n``, ``method``.

    References
    ----------
    Horowitz, J. L. (2009). *Semiparametric and Nonparametric Methods
    in Econometrics*. Springer, Sec. 6.3.1, eqs. (6.63)-(6.66),
    pp. 218-219.
    Horowitz, J. L. (1996). Semiparametric estimation of a regression
    model with an unknown transformation of the dependent variable.
    *Econometrica* 64(1), 103-137.
    """
    base = horowitz_transformation_model(x, y, ny=ny, nz=nz,
                                         bandwidth=bandwidth).payload
    T = base["T_hat"]
    ygrid = base["y_grid"]
    Z = base["index"]
    y2 = base["y2"]
    y1 = base["y1"]
    n = base["n"]
    nu = int(nu)
    if nu < 3:
        raise ValueError(f"nu must be at least 3, got {nu}.")

    Ty2 = T[0]
    Ty1 = T[len(T) - 1]
    yl = [float(t) for t in np.asarray(y, dtype=float).ravel()]
    U = [_tn_at(yl[i], ygrid, T, y2, y1) - Z[i] for i in range(n)]

    fin = [t for t in U if abs(t) < _BIG / 2.0]
    if len(fin) < 2:
        raise ValueError(
            "no observation of Y falls inside [y2, y1], so F is not "
            "estimable; widen the interval.")
    ulo = min(fin)
    uhi = max(fin)
    if uhi <= ulo:
        raise ValueError("the residuals U have no spread.")
    du = (uhi - ulo) / (nu - 1)
    ugrid = [ulo + k * du for k in range(nu)]
    # Pin the endpoints exactly.  The accumulated value ulo + (nu-1)*du
    # need not round to uhi, and F is an INDICATOR count: a last grid
    # point a single ulp below max(U) excludes that observation and
    # drops F(u_max) from 1 to 1 - 1/n.
    ugrid[0] = ulo
    ugrid[nu - 1] = uhi

    A = [0.0] * nu
    B = [0.0] * nu
    F = [0.0] * nu
    for k in range(nu):
        u = ugrid[k]
        lo = Ty2 - u
        hi = Ty1 - u
        a = b = 0.0
        for i in range(n):
            inb = 1.0 if (Z[i] > lo and Z[i] <= hi) else 0.0
            b += inb
            if inb > 0.0 and U[i] <= u:
                a += 1.0
        A[k] = a / n
        B[k] = b / n
        F[k] = (a / b) if b > 0 else float("nan")

    a_le_b = True
    f_unit = True
    for k in range(nu):
        if A[k] > B[k] + 1e-12:
            a_le_b = False
        if B[k] > 0 and not (-1e-12 <= F[k] <= 1.0 + 1e-12):
            f_unit = False

    return RichResult(payload={
        "T_hat": T,
        "F_hat": F,
        "beta_hat": base["beta_hat"],
        "u_grid": ugrid,
        "A_n": A,
        "B_n": B,
        "y_grid": ygrid,
        "y0": base["y0"],
        "index": Z,
        "F_in_unit": f_unit,
        "A_le_B": a_le_b,
        "n": n,
        "method": "Horowitz (2009) eqs. (6.60) and (6.66), T and F both nonparametric",
    })


def cheatsheet():
    return "hrztf: (6.66) F = An/Bn; the empirical CDF of Un is INCONSISTENT (p.219)"
