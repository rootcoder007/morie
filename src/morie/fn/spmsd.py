"""Mean-square differentiability of a random field."""

import numpy as np

from ._richresult import RichResult

__all__ = ["schabenberger_mean_square_diff"]


def schabenberger_mean_square_diff(cov_func, m=1, h=1e-3, tol=1e6):
    r"""
    Mean-square differentiability, from the even derivatives of C at 0.

    Stein (1999, Ch. 2.6), quoted by the book: :math:`Z(s)` is
    :math:`m`-times mean-square differentiable IF AND ONLY IF

    .. math::  \left.\frac{d^{2m}C(h)}{dh^{2m}}\right|_{0}

    exists and is finite. The covariance function of the :math:`m`-th
    derivative field is then :math:`(-1)^m d^{2m}C/dh^{2m}`.

    This is what separates the standard models. The gaussian covariance
    (eq. 2.6) is infinitely differentiable -- the book notes Stein regards
    that degree of smoothness as unrealistic for physical processes. The
    exponential model has a kink at the origin, so no second derivative
    exists and it is not mean-square differentiable at all.

    The even derivative is estimated by a central difference on a
    symmetric stencil, which is exact for the smooth case and blows up
    for the kinked one -- the divergence IS the diagnostic, so ``tol``
    is a magnitude ceiling rather than an accuracy target.

    Parameters
    ----------
    cov_func : callable
        ``C(h)``, taking an array of lags and returning an array.
    m : int, default 1
        Order of differentiability to test.
    h : float, default 1e-3
        Stencil spacing.
    tol : float, default 1e6
        Magnitude above which the derivative is judged not finite.

    Returns
    -------
    RichResult
        ``is_differentiable``, ``order``, ``derivative_2m`` (the estimate
        of :math:`d^{2m}C/dh^{2m}` at 0), ``derivative_cov`` (the implied
        covariance of the m-th derivative field, :math:`(-1)^m` times it).

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Sec. 2.3, pp. 50-51,
    citing Stein (1999), Ch. 2.6.
    """
    from math import comb

    if not callable(cov_func):
        raise TypeError("`cov_func` must be callable, C(h) -> array")
    m = int(m)
    if m < 1:
        raise ValueError("`m` must be >= 1")
    if h <= 0:
        raise ValueError("`h` must be > 0")

    k = 2 * m
    offs = np.arange(k + 1) - k // 2
    coeffs = np.array([(-1) ** i * comb(k, i) for i in range(k + 1)], dtype=float)

    def deriv_at(step):
        vals = np.asarray(cov_func(np.abs(offs[::-1] * step)), dtype=float).ravel()
        return float(np.sum(coeffs * vals) / step**k)

    # The criterion is EXISTENCE of the derivative, so the test is whether
    # the finite difference CONVERGES as the stencil shrinks -- not whether
    # it happens to be below some ceiling. For a kinked C the estimate
    # diverges like a power of 1/h, so halving h inflates it; for a smooth
    # C it settles. A magnitude ceiling gets this wrong: the exponential
    # model at h = 1e-3 gives about -6e3, comfortably under any generous
    # bound, yet has no second derivative at all.
    d1, d2 = deriv_at(h), deriv_at(h / 2.0)
    growth = abs(d2) / max(abs(d1), 1e-300)
    converged = bool(np.isfinite(d1) and np.isfinite(d2) and growth < 1.5)
    finite = bool(converged and abs(d2) <= tol)
    return RichResult(
        title="Mean-square differentiability",
        summary_lines=[("order m", m), ("d^{2m}C/dh^{2m} at 0", d2),
                       ("growth on halving h", growth),
                       ("differentiable", finite)],
        payload={"is_differentiable": finite, "order": m,
                 "derivative_2m": d2, "derivative_coarse": d1,
                 "growth_ratio": float(growth), "converged": converged,
                 "derivative_cov": float((-1) ** m * d2), "h": float(h)},
    )


def cheatsheet():
    return "spmsd: m-times MS differentiable iff d^2m C/dh^2m at 0 is finite."
