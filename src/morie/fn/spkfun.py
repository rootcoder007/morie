"""Ripley's K-function: expected number of extra events within distance h."""

from . import _array_core as np

from ._richresult import RichResult
from ._schab_pp import as_region, intensity, k_function

__all__ = ["schabenberger_k_function"]


def schabenberger_k_function(points, lambda_est=None, r=None, region=None,
                             correction="border"):
    r"""
    Ripley's K-function.

    .. math::

        K(h) = \frac{2\pi}{\lambda^2} \int_0^h x \lambda_2(x)\,dx

    For a simple process :math:`\lambda K(h)` is the expected number of
    EXTRA events within distance :math:`h` of an arbitrary event. Under
    the homogeneous Poisson process that expectation is
    :math:`\lambda \pi h^2`, so

    .. math::

        K_{\mathrm{HPP}}(h) = \pi h^2

    which is the reference curve every CSR comparison is made against.
    Clustering shows as :math:`K(h) > \pi h^2` at short lags, regularity
    as :math:`K(h) < \pi h^2`.

    Estimation follows Sec. 3.4.2: :math:`\hat\lambda = N(A)/\nu(A)`
    and :math:`\tilde{E}(h) = n^{-1}\sum_i\sum_{j \ne i} I(h_{ij} \le h)`,
    with :math:`\hat{K}(h) = \hat\lambda^{-1}\hat{E}(h)`. The naive form
    is negatively biased because events outside the window are not seen;
    the default border correction keeps only events further than
    :math:`h` from the boundary.

    Parameters
    ----------
    points : array-like
        Event coordinates, shape ``(n, 2)``.
    lambda_est : float, optional
        Intensity. Estimated by eq. (3.8) from ``region`` when omitted.
    r : array-like, optional
        Distances at which to evaluate. Defaults to 20 values out to a
        quarter of the smaller region side.
    region : array-like, optional
        ``(xmin, ymin, xmax, ymax)`` or vertices. Defaults to the
        bounding box of ``points``.
    correction : {'border', 'none'}
        Edge correction. ``'none'`` is the naive, negatively biased form.

    Returns
    -------
    RichResult
        ``r``, ``k`` (estimate), ``k_csr`` (:math:`\pi r^2`),
        ``lambda_est``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Secs. 3.4.1-3.4.2,
    eqs. (3.7)-(3.8), pp. 101-102.
    """
    reg = as_region(region, points)
    if r is None:
        side = min(reg[2] - reg[0], reg[3] - reg[1])
        r = np.linspace(0.0, side / 4.0, 20)
    r = np.atleast_1d(np.asarray(r, dtype=float))
    k = k_function(points, reg, r, correction)
    if lambda_est is not None:
        # rescale from the estimated intensity to the supplied one
        k = k * (intensity(points, reg) / float(lambda_est))
    lam = float(lambda_est) if lambda_est is not None else intensity(points, reg)
    return RichResult(
        title="Ripley's K-function",
        summary_lines=[("lambda", lam), ("correction", correction)],
        payload={"r": r, "k": k, "k_csr": np.pi * r**2,
                 "lambda_est": lam, "correction": correction},
    )


def cheatsheet():
    return "spkfun: Ripley's K; K(h)=pi h^2 under CSR."
