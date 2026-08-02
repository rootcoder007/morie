# SPDX-License-Identifier: AGPL-3.0-or-later
"""GWR kernel weight functions: Gaussian, bisquare, tricube, boxcar."""

from . import _array_core as np

from ._richresult import RichResult
from ._schab_gwr import KERNELS, adaptive_bandwidth, kernel_weights

__all__ = ["schabenberger_gwr_kernels"]


def schabenberger_gwr_kernels(distance, bandwidth, kernel_type="gaussian",
                              adaptive=False, normalized=False):
    """Geographical weights ``w_i(u)`` for a GWR kernel.

    The kernel decides how fast an observation's influence on the local fit
    at ``u`` decays with distance. Schabenberger & Gotway require only that
    "the influence of an observation decreases with increasing distance from
    s0" (Sec. 6.1.3.1, p. 317) and refer the reader to the kernels of
    Sec. 5.3.2 and to Fotheringham et al. (2002); the four kernels named by
    this module come from the GWR literature rather than from the book.

    ================  ====================================  ==============
    kernel            w(d)                                  support
    ================  ====================================  ==============
    ``gaussian``      ``exp(-0.5 (d/h)^2)``                 all d
    ``bisquare``      ``(1 - (d/h)^2)^2``                   ``d < h``
    ``tricube``       ``(1 - (d/h)^3)^3``                   ``d < h``
    ``boxcar``        ``1``                                 ``d < h``
    ================  ====================================  ==============

    The three truncated kernels are zero at and beyond ``h``. Bisquare and
    tricube also reach zero *smoothly* -- their first derivative vanishes at
    the edge -- which is what the GWR white paper means by calling the
    bisquare "near-Gaussian"; the boxcar drops discontinuously and is the
    only one of the four that does.

    Sec. 5.3.2 writes its Gaussian as a probability density,
    ``(1/(h sqrt(2 pi))) exp(-0.5 (d/h)^2)``, where the GWR literature drops
    the leading constant. Both are offered here (``normalized``) and they
    are not in conflict: weighted least squares is invariant to a positive
    scalar applied to every weight, so the two produce an identical hat
    matrix and identical local coefficients. Only the printed weights
    differ.

    Parameters
    ----------
    distance : array-like
        Distances ``d_i(u)`` from the regression point. Must be
        non-negative.
    bandwidth : float or int
        With ``adaptive=False``, the bandwidth ``h`` in coordinate units.
        With ``adaptive=True``, a neighbour count: ``h`` becomes the
        distance to the ``bandwidth``-th nearest point, so the same number
        of observations enters every local fit however unevenly the sample
        is spread. The regression point counts as its own first neighbour.
    kernel_type : {'gaussian', 'bisquare', 'tricube', 'boxcar'}
    adaptive : bool, default False
    normalized : bool, default False
        Return Sec. 5.3.2's density form. Gaussian only.

    Returns
    -------
    RichResult
        Keys: ``weights``, ``bandwidth`` (the distance actually used),
        ``kernel``, ``adaptive``, ``n_nonzero``, ``truncated``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Sec. 5.3.2, pp. 240-241
    (Epanechnikov and Gaussian kernels; the tri-cube attributed to
    Cleveland); Sec. 6.1.3.1, p. 317.
    Cleveland, W. S. (1979). Robust locally weighted regression and
    smoothing scatterplots. Journal of the American Statistical
    Association, 74:829-836. The tri-cube.
    Charlton, M. Geographically Weighted Regression -- White Paper, pp. 6-7
    (Gaussian and bisquare, with the adaptive-bandwidth rule).
    Fotheringham, A. S., Brunsdon, C. & Charlton, M. E. (2002).
    Geographically Weighted Regression. Wiley, New York, pp. 56-57, as
    cited by the ``GWmodel`` R package, which is the source consulted for
    the boxcar.
    """
    d = np.asarray(distance, dtype=float)
    if adaptive:
        h = adaptive_bandwidth(d, bandwidth)
    else:
        h = float(bandwidth)
    w = kernel_weights(d, h, kernel_type, normalized=normalized)
    truncated = kernel_type != "gaussian"
    payload = {
        "weights": w,
        "bandwidth": float(h),
        "kernel": kernel_type,
        "adaptive": bool(adaptive),
        "normalized": bool(normalized),
        "truncated": truncated,
        "n_nonzero": int(np.sum(np.asarray(w) > 0)),
    }
    return RichResult(
        title=f"GWR kernel weights ({kernel_type})",
        summary_lines=[("kernel", kernel_type), ("bandwidth", h),
                       ("truncated", truncated),
                       ("non-zero weights", payload["n_nonzero"])],
        payload=payload,
    )


def cheatsheet():
    return ("spgwrk: GWR kernel weights -- " + ", ".join(KERNELS) +
            "; fixed or adaptive (nearest-neighbour) bandwidth")
