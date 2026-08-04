# morie.fn -- function file (rootcoder007/morie)
"""ARE as the ratio of squared efficacies -- Theorem 13.2.2."""

import math

from ._richresult import RichResult

__all__ = ['areratio', 'gibbons_are_formula']


def areratio(deriv, var, deriv_star, var_star):
    """ARE(T, T*) from the two derivatives and the two null variances.

    Theorem 13.2.2 (book p. 485), eq. (13.2.1):

    .. math:: ARE(T, T^*) = \\lim_{n\\to\\infty}
        \\left[\\frac{dE(T_n)/d\\theta}
                   {dE(T_n^*)/d\\theta}\\right]^2
        \\frac{\\sigma^2(T_n^*)}{\\sigma^2(T_n)},

    all evaluated at theta = theta_0.  Because eq. (13.2.4) defines the
    efficacy as e(T) = [dE(T)/dtheta]^2 / sigma^2(T), this is exactly
    the ratio of efficacies e(T)/e(T*), which is returned as
    ``check`` so the two routes agree by construction.

    Parameters
    ----------
    deriv, var : float
        dE(T_n)/dtheta and sigma^2(T_n) for the first test.
    deriv_star, var_star : float
        The same two quantities for the reference test.

    Returns
    -------
    RichResult
        keys ``are``, ``check`` (efficacy ratio), ``efficacy``,
        ``efficacy_star``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Theorem 13.2.2, eq. (13.2.1), p. 485.
    """
    d = float(deriv)
    v = float(var)
    ds = float(deriv_star)
    vs = float(var_star)
    if v <= 0.0 or vs <= 0.0:
        raise ValueError("variances must be strictly positive.")
    if ds == 0.0:
        raise ValueError("the reference derivative must be non-zero.")
    are = (d / ds) ** 2 * vs / v
    e1 = d * d / v
    e2 = ds * ds / vs
    return RichResult(
        payload={
            "are": float(are),
            "check": float(e1 / e2),
            "efficacy": float(e1),
            "efficacy_star": float(e2),
            "method": "ARE = (dE ratio)^2 * var ratio, eq. (13.2.1)",
        }
    )


gibbons_are_formula = areratio
