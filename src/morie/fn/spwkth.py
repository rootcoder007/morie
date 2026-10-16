"""Wiener-Khinchin: covariance and spectral density are a transform pair."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["schabenberger_wiener_khinchin"]


def schabenberger_wiener_khinchin(cov_func, omega=None, h_max=200.0, n=40001):
    r"""
    Spectral density of a stationary field from its covariance function.

    Building the field as a sum of sinusoids with random amplitudes and
    uniform random phases gives (eq. 2.27)

    .. math::  C(h) = \sum_j \sigma_j^2 \cos(\omega_j h)

    and in the limit as the frequency spacing goes to zero

    .. math::  C(h) = \int_{-\infty}^{\infty} \cos(\omega h)\, s(\omega)\, d\omega

    with the inverse

    .. math::  s(\omega) = \frac{1}{2\pi}\int_{-\infty}^{\infty}
               \cos(\omega h)\, C(h)\, dh

    Both C and s are even, so only cosine terms survive.

    Parameters
    ----------
    cov_func : callable
        ``C(h)`` on the line, taking and returning arrays.
    omega : array-like, optional
        Frequencies at which to return the density.
    h_max, n : float, int
        Half-width and node count of the quadrature grid. Both integrals
        are numerical, so the outputs carry quadrature error, not just
        rounding: a slowly-decaying C needs a wide h grid and a
        heavy-tailed s needs a wide omega grid. ``integrated_density``
        is returned precisely so that error is visible -- it must equal
        C(0), and the shortfall measures the tail mass truncated away.

    Returns
    -------
    RichResult
        ``omega``, ``spectral_density``, ``variance`` (:math:`C(0)`),
        ``integrated_density`` (which must equal the variance).

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Sec. 2.5.3, eq. (2.27),
    pp. 66-68.
    """
    if not callable(cov_func):
        raise TypeError("`cov_func` must be callable, C(h) -> array")
    if omega is None:
        omega = np.linspace(0.0, 10.0, 201)
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    h = np.linspace(-h_max, h_max, int(n))
    ch = np.asarray(cov_func(np.abs(h)), dtype=float)
    s = np.array([np.trapezoid(np.cos(w * h) * ch, h) / (2 * np.pi) for w in omega])
    var = float(np.asarray(cov_func(np.zeros(1))).ravel()[0])
    # The omega grid cannot outrun the h grid. cos(omega h) sampled at
    # spacing dh aliases above the Nyquist frequency pi / dh, and
    # integrating past it sums noise rather than tail mass -- widening
    # the range there makes the answer worse, not better.
    dh = float(h[1] - h[0])
    w_nyq = 0.5 * np.pi / dh
    # integral of s over [-w_nyq, w_nyq] by the trapezoid rule on a
    # 20001-node omega grid. Both rules are linear, so the order of the
    # two sums can be swapped, and the omega rule applied to cos(omega h)
    # has a closed form (a Dirichlet kernel):
    #   sum_k cos(w_k h) = sin(K d h / 2) / sin(d h / 2),  K nodes, step d,
    # minus half of each endpoint term. That is the same discrete double
    # sum, but O(n) rather than the O(n * 20001) = 8e8 cosines the
    # explicit 20001 x 40001 table took -- many minutes on the array core.
    K = 20001
    dw = 2.0 * w_nyq / (K - 1)
    half = 0.5 * dw * h
    sh = np.sin(half)
    dirichlet = np.where(sh == 0, float(K), np.sin(K * half) / np.where(sh == 0, 1.0, sh))
    rule_w = dw * (dirichlet - np.cos(w_nyq * h))
    integrated = float(np.trapezoid(rule_w * ch, h) / (2 * np.pi))
    return RichResult(
        title="Wiener-Khinchin pair",
        summary_lines=[
            ("C(0)", var),
            ("integral of s", integrated),
            ("Nyquist omega", float(w_nyq)),
        ],
        payload={
            "omega": omega,
            "spectral_density": s,
            "variance": var,
            "integrated_density": integrated,
            "nyquist_omega": float(w_nyq),
        },
    )


def cheatsheet():
    return "spwkth: s(w)=(1/2pi) int cos(wh)C(h)dh; C(0)=int s(w)dw."
