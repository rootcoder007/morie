"""Matern covariance function class."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["schabenberger_matern_covariance"]


def schabenberger_matern_covariance(h, sigma2=1.0, nu=0.5, a=1.0):
    r"""
    Matern class of covariance functions.

    .. math::

        C(h) = \sigma^2 \frac{1}{\Gamma(\nu)}
               \left(\frac{\theta h}{2}\right)^{\nu} 2 K_{\nu}(\theta h),
        \qquad \nu > 0,\; \theta > 0

    :math:`\theta` governs the range; smoothness increases with
    :math:`\nu`. Because :math:`K_\nu(t) \approx \tfrac{\Gamma(\nu)}{2}
    (t/2)^{-\nu}` as :math:`t \to 0`, the variance of the process is
    exactly :math:`\sigma^2`, so ``C(0) = sigma2``.

    The book names three members: :math:`\nu = 1/2` is the exponential
    model (eq. 4.11), :math:`\nu = 1` is Whittle's model (eq. 4.12), and
    :math:`\nu \to \infty` is the gaussian model (eq. 4.10).

    Note the parameterisation. Here ``a`` is the book's :math:`\theta`,
    the scale in eq. (4.9) -- NOT the practical range. The practical
    range of a Matern model is itself a function of :math:`\nu`
    (p. 143), which is why the book gives the simple practical-range
    forms only for the named special cases.

    Parameters
    ----------
    h : array-like
        Lag distances, non-negative.
    sigma2 : float, default 1.0
        Process variance :math:`\sigma^2`, must be positive.
    nu : float, default 0.5
        Smoothness :math:`\nu > 0`.
    a : float, default 1.0
        Scale :math:`\theta > 0`.

    Returns
    -------
    RichResult
        ``covariance``, ``semivariogram`` (:math:`\sigma^2 - C(h)`),
        ``sigma2``, ``nu``, ``theta``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Sec. 4.3.2, eq. (4.9),
    p. 143.
    """
    from math import gamma as _gammafn

    from scipy.special import kv

    if nu <= 0:
        raise ValueError("`nu` must be > 0 for the Matern class")
    if a <= 0:
        raise ValueError("`a` (theta) must be > 0")
    if sigma2 <= 0:
        raise ValueError("`sigma2` must be > 0")
    h = np.atleast_1d(np.asarray(h, dtype=float))
    if np.any(h < 0):
        raise ValueError("lag distances `h` must be non-negative")

    t = a * h
    c = np.empty_like(t)
    pos = t > 0
    c[~pos] = sigma2                       # limit at the origin is the variance
    tp = t[pos]
    c[pos] = sigma2 * (1.0 / _gammafn(nu)) * (tp / 2.0) ** nu * 2.0 * kv(nu, tp)
    return RichResult(
        title="Matern covariance",
        summary_lines=[("sigma^2", sigma2), ("nu", nu), ("theta", a)],
        payload={"covariance": c, "semivariogram": sigma2 - c,
                 "sigma2": float(sigma2), "nu": float(nu), "theta": float(a)},
    )


def cheatsheet():
    return "spmatr: Matern covariance, eq (4.9); nu=1/2 exponential, nu=1 Whittle."
