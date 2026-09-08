"""Besag's L function on a bare coordinate set."""

from __future__ import annotations

from ._richresult import RichResult

__all__ = ["ripley_l"]


def ripley_l(coords, r_grid=None, correction="border"):
    r"""Besag's L function, :math:`L(r) = \sqrt{K(r)/\pi} - r`.

    The same transform as :func:`morie.fn.ripL.ripley_l_function`, taking
    only coordinates: the observation window is the bounding box of
    ``coords``. Use ``ripL`` when the window is known and is not the
    bounding box, since the bounding box overstates the intensity of a
    pattern observed in a larger region and biases K downwards.

    The previous body was a placeholder: it averaged ``coords`` and never
    used ``r_grid``.

    Parameters
    ----------
    coords : array-like
        Event coordinates, shape ``(n, 2)``.
    r_grid : array-like, optional
        Distances at which to evaluate.
    correction : {'border', 'none'}
        Edge correction.

    Returns
    -------
    RichResult
        ``r``, ``l``, ``l_uncentred``, ``k``, ``lambda_est``, ``method``.

    References
    ----------
    Besag, J. (1977). Discussion of "Modelling spatial patterns" by
    B. D. Ripley. *Journal of the Royal Statistical Society B*, 39(2),
    193-195.

    Baddeley, A. & Turner, R. (2005). *Journal of Statistical Software*,
    12(6), p. 17.
    """
    from .ripL import ripley_l_function

    res = ripley_l_function(coords, None, r_grid, correction)
    return RichResult(
        title="Besag L function (bounding-box window)",
        summary_lines=[("lambda", res["lambda_est"])],
        payload={
            "r": res["r"],
            "l": res["l"],
            "l_uncentred": res["l_uncentred"],
            "k": res["k"],
            "lambda_est": res["lambda_est"],
            "method": "Besag L function on the bounding box of the coordinates",
        },
    )


def cheatsheet():
    return "rklfunc: L(r) = sqrt(K(r)/pi) - r, window = bounding box of coords."
