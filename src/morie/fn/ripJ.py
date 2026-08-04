"""van Lieshout & Baddeley's J function."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult
from ._schab_pp import as_points, as_region, nn_distances

__all__ = ["ripley_j_function"]


def ripley_j_function(points, window=None, r=None, n_grid=40):
    r"""The J function, :math:`J(r) = \{1 - G(r)\} / \{1 - F(r)\}`.

    :math:`G` is the event-to-event nearest neighbour distance CDF and
    :math:`F` the point-to-event empty space CDF. Under complete spatial
    randomness the two coincide, so :math:`J \equiv 1`; :math:`J < 1`
    indicates clustering and :math:`J > 1` regularity. Because both
    factors carry the same CSR form the ratio cancels the intensity,
    which is what makes J useful when :math:`\lambda` is poorly known.

    The previous body was a placeholder: it averaged the leading
    ``points`` argument and used neither ``window`` nor ``r``.

    Parameters
    ----------
    points : array-like
        Event coordinates, shape ``(n, 2)``.
    window : array-like, optional
        ``(xmin, ymin, xmax, ymax)`` or vertices; defaults to the bounding
        box of ``points``.
    r : array-like, optional
        Distances at which to evaluate. Defaults to 25 points from 0 to
        the largest nearest-neighbour distance.
    n_grid : int, default 40
        Side of the lattice used for the F estimate.

    Returns
    -------
    RichResult
        ``r``, ``j``, ``g``, ``f``, ``j_csr`` (a vector of ones),
        ``lambda_est``, ``n_defined``, ``method``.

    Notes
    -----
    :math:`J` is undefined once :math:`F(r) = 1`, i.e. beyond the largest
    empty-space distance in the sample; those entries are returned as NaN
    rather than as a division-by-zero infinity, and ``n_defined`` counts
    the finite ones.

    References
    ----------
    van Lieshout, M. N. M. & Baddeley, A. J. (1996). A nonparametric
    measure of spatial interaction in point patterns. *Statistica
    Neerlandica*, 50(3), 344-361. doi:10.1111/j.1467-9574.1996.tb01501.x

    Baddeley, A. & Turner, R. (2005). spatstat: an R package for analyzing
    spatial point patterns. *Journal of Statistical Software*, 12(6),
    p. 16, which states "J(r), the function J = (1 - G)/(1 - F)", and
    p. 17, attributing J to van Lieshout & Baddeley (1996).
    """
    from .spffun import schabenberger_f_function
    from .spgfun import schabenberger_g_function

    p = as_points(points)
    reg = as_region(window, p)
    if r is None:
        nn = nn_distances(p)
        if nn.size == 0:
            raise ValueError("at least two events are needed for the J function")
        r = np.linspace(0.0, float(nn.max()), 25)
    r = np.atleast_1d(np.asarray(r, dtype=float))

    gr = schabenberger_g_function(p, r, reg)
    fr = schabenberger_f_function(p, reg, r, n_grid)
    g = np.asarray(gr["g"], dtype=float)
    f = np.asarray(fr["f"], dtype=float)

    denom = 1.0 - f
    j = np.array(
        [float("nan") if denom[i] <= 0.0 else (1.0 - g[i]) / denom[i]
         for i in range(len(r))],
        dtype=float,
    )
    n_defined = int(sum(1 for v in j if v == v))

    return RichResult(
        title="J function (1-G)/(1-F)",
        summary_lines=[("lambda", gr["lambda_est"]), ("defined at", n_defined)],
        payload={
            "r": r,
            "j": j,
            "g": g,
            "f": f,
            "j_csr": np.ones(len(r)),
            "lambda_est": gr["lambda_est"],
            "n_defined": n_defined,
            "method": "J function, (1 - G(r)) / (1 - F(r))",
        },
    )


def cheatsheet():
    return "ripJ: J(r) = (1-G(r))/(1-F(r)); J = 1 under CSR, < 1 clustered, > 1 regular."
