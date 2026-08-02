"""Convolution representation of a stationary random field."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["schabenberger_convolution_representation"]


def schabenberger_convolution_representation(kernel=None, h=None, sigma2_x=1.0,
                                             half_width=5.0, n=40001):
    r"""
    A field built by convolving white noise with a kernel.

    Convolving a kernel :math:`K` with a white-noise field gives a
    second-order stationary process whose covariance is the convolution
    of the kernel with itself,

    .. math::

        \mathrm{Cov}[Z(s), Z(s+h)] = \sigma_x^2 \int K(u)K(u+h)\,du

    The construction is useful because ANY kernel yields a valid
    covariance -- positive-definiteness is automatic rather than
    something to check.

    The book's own worked case: a uniform (boxcar) kernel on the line
    convolves to a TENT correlation function, which is also the
    :math:`d = 1` member of the spherical family.

    Parameters
    ----------
    kernel : callable, optional
        ``K(u)``. Defaults to the boxcar on ``[-1/2, 1/2]``.
    h : array-like, optional
        Lags at which to evaluate the covariance.
    sigma2_x : float
        White-noise variance.
    half_width, n : float, int
        Quadrature half-width and node count.

    Returns
    -------
    RichResult
        ``h``, ``covariance``, ``correlation``, ``variance``
        (:math:`C(0)`).

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Sec. 2.4.2; the boxcar
    case appears again at Sec. 4.3.3, p. 146.
    """
    if kernel is None:
        def kernel(u):
            return (np.abs(np.asarray(u, dtype=float)) <= 0.5).astype(float)
    if not callable(kernel):
        raise TypeError("`kernel` must be callable, K(u) -> array")
    if sigma2_x <= 0:
        raise ValueError("`sigma2_x` must be > 0")
    if h is None:
        h = np.linspace(0.0, 2.0, 41)
    h = np.atleast_1d(np.asarray(h, dtype=float))
    u = np.linspace(-half_width, half_width, int(n))
    ku = np.asarray(kernel(u), dtype=float)
    cov = np.array([sigma2_x * np.trapezoid(ku * np.asarray(kernel(u + hh),
                                                            dtype=float), u)
                    for hh in h])
    c0 = float(sigma2_x * np.trapezoid(ku * ku, u))
    return RichResult(
        title="Convolution representation",
        summary_lines=[("C(0)", c0), ("lags", int(h.size))],
        payload={"h": h, "covariance": cov,
                 "correlation": cov / c0 if c0 > 0 else cov * np.nan,
                 "variance": c0},
    )


def cheatsheet():
    return "spconv: C(h)=sigma^2 int K(u)K(u+h)du; boxcar -> tent."
