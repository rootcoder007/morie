"""Empty space function F(r) on a fixed sample grid."""

from __future__ import annotations

from ._richresult import RichResult

__all__ = ["empty_space_function"]


def empty_space_function(coords, r_grid=None, window=None, n_grid=40):
    r"""Empty space function: the point-to-nearest-event distance CDF.

    :math:`F(r)` is the probability that the distance from an ARBITRARY
    location to the nearest event is at most :math:`r` -- the "contact
    distribution" or point-to-event distribution, as distinct from
    :math:`G`, which measures event-to-event. It is estimated from a
    deterministic ``n_grid`` by ``n_grid`` lattice of sample locations
    over the window, so repeated calls return the same numbers.

    Under complete spatial randomness with intensity :math:`\lambda`,

    .. math::
        F(r) = 1 - e^{-\lambda \pi r^2},

    the same form as :math:`G`, but the two move in opposite directions
    under departures: clustering leaves large empty gaps, pushing
    :math:`\hat F` BELOW the CSR curve while :math:`\hat G` rises above it.

    The previous body was a placeholder: it averaged ``coords`` and used
    neither ``r_grid`` nor ``window``.

    Parameters
    ----------
    coords : array-like
        Event coordinates, shape ``(n, 2)``.
    r_grid : array-like, optional
        Distances at which to evaluate the CDF.
    window : array-like, optional
        ``(xmin, ymin, xmax, ymax)`` or vertices; defaults to the bounding
        box of ``coords``.
    n_grid : int, default 40
        Side of the square lattice of sample locations.

    Returns
    -------
    RichResult
        ``r``, ``f``, ``f_csr``, ``empty_space_distances``,
        ``lambda_est``, ``n_sample``, ``method``.

    References
    ----------
    Baddeley, A. & Turner, R. (2005). spatstat: an R package for analyzing
    spatial point patterns. *Journal of Statistical Software*, 12(6),
    p. 16: "F(r), the empty space function (contact distribution or
    'point-to-event' distribution)".

    Schabenberger, O. & Gotway, C. A. (2005). *Statistical Methods for
    Spatial Data Analysis*. Chapman & Hall/CRC, sec. 3.3.4, pp. 97-98.
    """
    from .spffun import schabenberger_f_function

    res = schabenberger_f_function(coords, window, r_grid, n_grid)
    d = res["empty_space_distances"]
    return RichResult(
        title="Empty space function F(r)",
        summary_lines=[("sample locations", int(len(d))),
                       ("lambda", res["lambda_est"])],
        payload={
            "r": res["r"],
            "f": res["f"],
            "f_csr": res["f_csr"],
            "empty_space_distances": d,
            "lambda_est": res["lambda_est"],
            "n_sample": int(len(d)),
            "method": "Empty space function F(r) on a fixed lattice of sample locations",
        },
    )


def cheatsheet():
    return "empfun: F(r) = P(distance from an arbitrary location to nearest event <= r)"
