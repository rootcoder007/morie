"""Spectral representation of a stationary random field."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["schabenberger_spectral_representation"]


def schabenberger_spectral_representation(h, sigma2=None, omega=None, mu=0.0,
                                          seed=None):
    r"""
    Sum-of-sinusoids representation, eq. (2.26)-(2.27).

    .. math::

        Z(s) = \mu + \sum_{j=-K}^{K} A_j \cos(\omega_j s + \phi_j)

    with random amplitudes :math:`A_j` and phases :math:`\phi_j` uniform
    on :math:`(0, 2\pi)`, all mutually independent. The :math:`Y_j` have
    zero mean because :math:`\int_0^{2\pi}\cos(a+\phi)d\phi = 0`, and the
    covariance collapses to a cosine sum

    .. math::  C(h) = \sum_j \sigma_j^2 \cos(\omega_j h),
               \qquad \sigma_j^2 = \tfrac{1}{2}E[A_j^2]

    so the process variance is distributed over discrete frequencies:
    :math:`\mathrm{Var}[Z(s)] = C(0) = \sum_j \sigma_j^2`. That identity
    is the content of the representation -- the spectrum is a budget for
    the variance.

    Parameters
    ----------
    h : array-like
        Lags at which to evaluate ``C``.
    sigma2 : array-like, optional
        Variance at each frequency. Defaults to a decaying set.
    omega : array-like, optional
        The frequencies, same length as ``sigma2``.
    mu : float
        Process mean, carried through to a realisation.
    seed : int, optional
        Seed for the realisation of amplitudes and phases.

    Returns
    -------
    RichResult
        ``h``, ``covariance``, ``variance`` (:math:`C(0)`),
        ``sum_sigma2`` (must equal it), ``omega``, ``sigma2``,
        ``realisation``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Sec. 2.5, eqs.
    (2.26)-(2.27), pp. 66-67.
    """
    h = np.atleast_1d(np.asarray(h, dtype=float))
    if omega is None:
        omega = np.linspace(0.2, 4.0, 12)
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    if sigma2 is None:
        sigma2 = np.exp(-omega)
    sigma2 = np.atleast_1d(np.asarray(sigma2, dtype=float))
    if sigma2.shape != omega.shape:
        raise ValueError("`sigma2` and `omega` must have the same length")
    if np.any(sigma2 < 0):
        raise ValueError("`sigma2` entries must be non-negative variances")

    cov = np.array([np.sum(sigma2 * np.cos(omega * hh)) for hh in h])
    rng = np.random.default_rng(seed)
    amp = np.sqrt(2.0 * sigma2)
    phase = rng.uniform(0.0, 2 * np.pi, omega.size)
    realisation = mu + np.array(
        [np.sum(amp * np.cos(omega * s + phase)) for s in h])
    return RichResult(
        title="Spectral representation",
        summary_lines=[("C(0)", float(np.sum(sigma2))),
                       ("frequencies", int(omega.size))],
        payload={"h": h, "covariance": cov, "variance": float(np.sum(sigma2)),
                 "sum_sigma2": float(np.sum(sigma2)), "omega": omega,
                 "sigma2": sigma2, "realisation": realisation},
    )


def cheatsheet():
    return "spspec: C(h)=sum sigma_j^2 cos(w_j h); C(0)=sum sigma_j^2."
