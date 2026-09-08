"""Besag's L function: the variance-stabilised K, centred at zero under CSR."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult
from ._schab_pp import as_region

__all__ = ["ripley_l_function"]


def ripley_l_function(points, window=None, r=None, correction="border"):
    r"""Besag's L function, :math:`L(d) = \sqrt{K(d)/\pi} - d`.

    Ripley's K rises like :math:`\pi d^2` and its sampling variance grows
    with it, which makes a K plot hard to read. Besag's square-root
    transform removes both problems: under complete spatial randomness
    :math:`K(d) = \pi d^2`, so :math:`\sqrt{K(d)/\pi} = d` and the
    centred curve returned here is identically zero. Positive values at
    short distances indicate clustering, negative values regularity.

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
        Distances at which to evaluate.
    correction : {'border', 'none'}
        Edge correction, passed through to the K function.

    Returns
    -------
    RichResult
        ``r``, ``l`` (the centred curve :math:`\sqrt{K/\pi} - r`),
        ``l_uncentred`` (:math:`\sqrt{K/\pi}`), ``k``, ``lambda_est``,
        ``method``.

    Notes
    -----
    ``l`` here is the CENTRED curve, which is what the docstring formula
    of this module asks for; :func:`morie.fn.splfun` returns the
    uncentred :math:`\sqrt{K/\pi}` under the name ``l`` and the centred
    one as ``l_minus_r``. They agree wherever K is defined. This module
    computes L from K itself rather than through ``splfun`` so that an
    undefined K (border correction with no events left at that distance)
    stays NaN instead of being clamped to zero by ``max(K, 0)`` -- with
    the bounding box as the window that happens for every r beyond the
    largest border distance.

    References
    ----------
    Besag, J. (1977). Discussion of "Modelling spatial patterns" by
    B. D. Ripley. *Journal of the Royal Statistical Society B*, 39(2),
    193-195. (Ripley's paper itself is 39(2), 172-192,
    doi:10.1111/j.2517-6161.1977.tb01615.x.)

    Baddeley, A. & Turner, R. (2005). spatstat: an R package for analyzing
    spatial point patterns. *Journal of Statistical Software*, 12(6),
    p. 17, which states :math:`L(r) = \sqrt{\hat K(r)/\pi}`.
    """
    from .spkfun import schabenberger_k_function

    reg = as_region(window, points)
    res = schabenberger_k_function(points, None, r, reg, correction)
    rr = res["r"]
    kk = res["k"]
    # NaN in K means the border correction retained no events at that
    # distance, i.e. K is undefined there. Carry that through rather than
    # letting max(K, 0) quietly turn an undefined value into zero -- with
    # the bounding box as the window this happens for every r beyond the
    # largest border distance.
    lu = [float("nan") if kk[i] != kk[i] else np.sqrt(max(kk[i], 0.0) / np.pi)
          for i in range(len(rr))]
    lc = [lu[i] - rr[i] for i in range(len(rr))]
    return RichResult(
        title="Besag L function (centred)",
        summary_lines=[("lambda", res["lambda_est"]), ("correction", correction)],
        payload={
            "r": rr,
            "l": lc,
            "l_uncentred": lu,
            "k": kk,
            "lambda_est": res["lambda_est"],
            "method": "Besag L function, sqrt(K(d)/pi) - d",
        },
    )


def cheatsheet():
    return "ripL: L(d) = sqrt(K(d)/pi) - d; identically 0 under CSR."
