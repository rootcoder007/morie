# morie.fn -- function file (rootcoder007/morie)
"""Chen gamma kernel density estimator for [0,inf) data."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["fauzi_gamma_kde"]


def fauzi_gamma_kde(x, grid=None, h=None, modified=False, a=2.0):
    r"""Chen's (1999) gamma kernel density estimator, and Fauzi's
    modified version (Fauzi Ch. 1):

    .. math:: \hat f(x) = \frac1n\sum_i K_{x/h+1,\,h}(X_i),

    with :math:`K_{a,b}` the Gamma(a, b) density.

    The kernel's SUPPORT is :math:`[0,\infty)` and its SHAPE varies
    with the evaluation point, so no mass ever crosses the boundary.
    That removes the O(h) boundary bias of a symmetric kernel
    entirely -- the estimator is consistent at zero, where
    :mod:`morie.fn.fzkde` is not.

    The modification the chapter proposes combines two gamma
    estimators with bandwidths :math:`h` and :math:`ah` by
    self-elimination:

    .. math:: \hat f_{mod}
              = \frac{a\,\hat f_h - \hat f_{ah}}{a - 1},

    which cancels the leading bias term while KEEPING the boundary
    property. It reduces variance relative to bias-corrected
    alternatives; ``bias_order`` records the improvement.

    Parameters
    ----------
    x : array-like
        Sample on [0, infinity).
    grid : array-like, optional
        Evaluation points, non-negative.
    h : float, optional
        Bandwidth.
    modified : bool
        Use the self-elimination combination.
    a : float
        The second bandwidth multiplier, not equal to 1.

    Returns
    -------
    RichResult
        keys: ``grid``, ``density``, ``bandwidth``, ``modified``,
        ``a``, ``boundary_consistent`` (True), ``bias_order``,
        ``mass``, ``n``, ``method``.
    """
    from ._fauzi import gamma_kernel_density

    xv = np.asarray(x, dtype=float).ravel()
    n = xv.size
    if n < 2:
        raise ValueError(f"need at least 2 observations, got {n}.")
    if np.any(xv < 0):
        raise ValueError("gamma kernels need data on [0, infinity).")
    hh = float(np.std(xv, ddof=1) * n ** -0.4) if h is None else float(h)
    if hh <= 0:
        raise ValueError(f"bandwidth must be positive, got {hh}.")
    g = np.linspace(0.0, float(xv.max()) * 1.2, 200) if grid is None else \
        np.atleast_1d(np.asarray(grid, dtype=float))
    base = gamma_kernel_density(xv, g, hh)
    if modified:
        av = float(a)
        if av == 1.0 or av <= 0:
            raise ValueError(f"a must be positive and not 1, got {av}.")
        second = gamma_kernel_density(xv, g, av * hh)
        dens = (av * base - second) / (av - 1.0)
        order = "o(h^2): the leading term is cancelled by self-elimination"
    else:
        dens = base
        order = "O(h)"
    return RichResult(payload={
        "grid": g, "density": dens, "bandwidth": hh,
        "modified": bool(modified), "a": float(a),
        "boundary_consistent": True, "bias_order": order,
        "mass": float(np.trapezoid(dens, g)),
        "why_it_works": "the kernel's support IS [0, infinity), so no mass "
                        "crosses the boundary and no correction is needed",
        "n": int(n),
        "method": "Chen gamma kernel density, with Fauzi's self-elimination modification"})


def cheatsheet():
    return "fzgkde: the kernel's support matches the data's -- boundary bias never arises"


# compact alias per ledger/NAMING.md
fauzigammakde = fauzi_gamma_kde
