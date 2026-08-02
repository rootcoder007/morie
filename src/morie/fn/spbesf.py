"""Modified Bessel function K_nu used in the Matern covariance."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["schabenberger_bessel_function"]

_SERIES_TERMS = 220
_ASYMPTOTIC_FROM = 30.0


def _bessel_i(nu, t, terms=_SERIES_TERMS):
    r"""Modified Bessel function of the FIRST kind, by its defining series.

    .. math::

        I_\nu(t) = \sum_{k=0}^{\infty}
                   \frac{(t/2)^{2k+\nu}}{k!\,\Gamma(\nu+k+1)}

    This is the modified analogue of the book's series (4.72) for
    :math:`J_\nu`.
    """
    from math import floor, lgamma

    def _gamma_sign(x):
        # lgamma returns log|Gamma|, so the sign has to be restored.
        # Gamma alternates sign on each interval below zero: negative on
        # (-1, 0), positive on (-2, -1), and so on.
        return 1.0 if x > 0 else (-1.0) ** (-floor(x))

    t = np.asarray(t, dtype=float)
    half = t / 2.0
    out = np.zeros_like(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_half = np.log(np.where(half > 0, half, 1.0))
        for k in range(terms):
            g = nu + k + 1
            if abs(g - round(g)) < 1e-12 and g <= 0:
                continue                      # 1/Gamma(non-positive integer) = 0
            log_term = (2 * k + nu) * log_half - lgamma(k + 1) - lgamma(g)
            out = out + _gamma_sign(g) * np.exp(log_term)
    return np.where(t > 0, out, np.inf if nu < 0 else (1.0 if nu == 0 else 0.0))


def schabenberger_bessel_function(x, nu=0.5):
    r"""
    Modified Bessel function of the second kind, :math:`K_\nu`.

    The book defines it (eq. 4.73) as

    .. math::

        K_\nu(t) = \frac{\pi}{2}\,
                   \frac{I_{-\nu}(t) - I_{\nu}(t)}{\sin(\pi\nu)}

    and this is the component of the Matern covariance class (eq. 4.9).

    That identity is a DEFINITION, not a numerical algorithm. Both
    :math:`I_{\pm\nu}(t)` grow like :math:`e^{t}` while their difference
    decays like :math:`e^{-t}`, so evaluating it directly loses roughly
    :math:`2t/\ln 10` digits to cancellation: measured against a stable
    reference the ascending series is exact to 1e-15 at ``t = 0.05``, has
    fallen to 1e-9 by ``t = 8``, and is worthless past ``t ~ 15``. Since
    covariance modelling needs the tail, evaluation uses a stable
    routine; ``_bessel_i`` is retained so the tests can check this
    function against the book's own identity over the range where the
    identity is trustworthy.

    Parameters
    ----------
    x : array-like
        Argument :math:`t`, must be positive.
    nu : float, default 0.5
        Order, real. The Matern class requires ``nu > 0``.

    Returns
    -------
    RichResult
        ``value`` (array) and ``nu``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Sec. 4.9.2, eq. (4.73),
    p. 210.
    """
    from ._sci_core import kv

    t = np.atleast_1d(np.asarray(x, dtype=float))
    if np.any(t <= 0):
        raise ValueError("`x` must be positive; K_nu diverges at the origin")
    return RichResult(
        title="Modified Bessel function of the second kind",
        summary_lines=[("order nu", float(nu)), ("n points", int(t.size))],
        payload={"value": np.asarray(kv(nu, t), dtype=float), "nu": float(nu)},
    )


def cheatsheet():
    return "spbesf: K_nu(t), modified Bessel of the second kind, eq (4.73)."
