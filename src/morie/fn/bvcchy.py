"""Bivariate Cauchy density."""

import math

from ._richresult import RichResult

__all__ = ["bivariate_cauchy_density"]


def bivariate_cauchy_density(z1, z2, c=1.0):
    r"""Density of the bivariate Cauchy distribution (Mardia 1970, p. 86).

    .. math::

        f(z_1, z_2) = \frac{c}{2\pi}\,(c^2 + z_1^2 + z_2^2)^{-3/2},
        \qquad c > 0,

    the bivariate t distribution with one degree of freedom and scale
    matrix :math:`c^2 I`. Both marginals are Cauchy with scale ``c``, yet
    :math:`E[Z_2 \mid Z_1]` does not exist -- the example Schabenberger &
    Gotway (2005, p. 293) use to show that specified marginals do not
    make a usable joint model.

    Parameters
    ----------
    z1, z2 : float or sequence of float
        Evaluation points, the same length.
    c : float
        Scale, positive.

    Returns
    -------
    RichResult
        ``density`` (list), ``c``.

    References
    ----------
    Mardia, K. V. (1970). Families of Bivariate Distributions. Hafner,
    p. 86. Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods
    for Spatial Data Analysis. Chapman & Hall/CRC, Sec. 5.8, p. 293.
    """
    c = float(c)
    if not c > 0:
        raise ValueError("`c` must be positive")
    a = [float(z1)] if isinstance(z1, (int, float)) else [float(v) for v in z1]
    b = [float(z2)] if isinstance(z2, (int, float)) else [float(v) for v in z2]
    if len(a) != len(b):
        raise ValueError("`z1` and `z2` must have the same length")
    dens = [c / (2.0 * math.pi) * (c * c + u * u + v * v) ** -1.5 for u, v in zip(a, b)]
    return RichResult(
        title="Bivariate Cauchy density",
        summary_lines=[("c", c), ("points", len(dens))],
        payload={"density": dens, "c": c},
    )


def cheatsheet():
    return "bvcchy: bivariate Cauchy density c/(2 pi) (c^2 + z1^2 + z2^2)^(-3/2)"
