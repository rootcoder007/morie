"""Joint density of X and Y = mX + Z (Gaussian X, Z): both forms of rho(x, y).

Morin (2016), Probability: For the Enthusiastic Beginner, eqs (6.28)-(6.34).
"""

import math

from ._richresult import RichResult

__all__ = ["bvnmodel"]


def bvnmodel(x, y, m, sigma_x, sigma_z):
    """rho(x, y) for Y = mX + Z with X ~ N(0, sigma_x^2), Z ~ N(0, sigma_z^2) independent.

    Model form (6.28)-(6.30): rho(x, y) = rho_X(x) rho_Z(y - m x)
    = exp(-x^2/2 sigma_x^2 - (y - m x)^2/2 sigma_z^2) / (2 pi sigma_x sigma_z).
    Correlation form (6.31)/(6.34), with sigma_y = sqrt(m^2 sigma_x^2 + sigma_z^2)
    and r = m sigma_x/sigma_y:
    exp(-[x^2/sigma_x^2 - 2 r x y/(sigma_x sigma_y) + y^2/sigma_y^2] / 2(1 - r^2))
    / (2 pi sigma_x sigma_y sqrt(1 - r^2)).

    Parameters
    ----------
    x, y : float
        Point.
    m : float
        Slope.
    sigma_x, sigma_z : float
        Standard deviations, > 0.

    Returns
    -------
    RichResult
        Keys: density (model form), density_r (correlation form), sigma_y, r.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eqs (6.28)-(6.34).

    Examples
    --------
    >>> round(bvnmodel(0.0, 0.0, 1.0, 1.0, 1.0)["density"] * 2 * math.pi, 12)
    1.0
    """
    x, y, m, sx, sz = map(float, (x, y, m, sigma_x, sigma_z))
    if sx <= 0 or sz <= 0:
        raise ValueError("sigma_x and sigma_z must be > 0")
    d = math.exp(-x * x / (2 * sx * sx) - (y - m * x) ** 2 / (2 * sz * sz)) / (2 * math.pi * sx * sz)
    sy = math.sqrt(m * m * sx * sx + sz * sz)
    r = m * sx / sy
    q = x * x / (sx * sx) - 2 * r * x * y / (sx * sy) + y * y / (sy * sy)
    dr = math.exp(-q / (2 * (1 - r * r))) / (2 * math.pi * sx * sy * math.sqrt(1 - r * r))
    return RichResult(
        title="Joint density of X and Y = mX + Z",
        summary_lines=[("rho(x, y)", d), ("r", r)],
        payload={"density": d, "density_r": dr, "sigma_y": sy, "r": r},
    )


def cheatsheet():
    return "bvnmodel: joint density of X and Y = mX + Z in model and correlation form. Morin (2016) eqs (6.28)-(6.34)."
