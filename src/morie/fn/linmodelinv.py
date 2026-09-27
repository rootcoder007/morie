"""Y = mX + Z model from (sigma_x, sigma_y, r): m = r sigma_y/sigma_x, sigma_z = sigma_y sqrt(1 - r^2).

Morin (2016), Probability: For the Enthusiastic Beginner, eqs (6.16)-(6.18), (6.35).
"""

import math

from ._richresult import RichResult

__all__ = ["linmodelinv"]


def linmodelinv(sigma_x, sigma_y, r):
    """Convert the (sigma_x, sigma_y, r) description to the (sigma_x, sigma_z, m) one.

    Inverting sigma_y = sqrt(m^2 sigma_x^2 + sigma_z^2) and r = m sigma_x / sigma_y
    gives m = r sigma_y / sigma_x and sigma_z = sigma_y sqrt(1 - r^2) (6.18), so
    Y = (r sigma_y/sigma_x) X + Z with Z independent of X (6.35).

    Parameters
    ----------
    sigma_x, sigma_y : float
        Standard deviations, > 0.
    r : float
        Correlation, -1 <= r <= 1.

    Returns
    -------
    RichResult
        Keys: m, sigma_z.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eqs (6.16)-(6.18), (6.35).

    Examples
    --------
    >>> r = linmodelinv(1.0, 2.0, 0.6)
    >>> round(r["m"], 12), round(r["sigma_z"], 12)
    (1.2, 1.6)
    """
    sx, sy, r = float(sigma_x), float(sigma_y), float(r)
    if sx <= 0 or sy <= 0 or not -1 <= r <= 1:
        raise ValueError("need sigma_x, sigma_y > 0 and -1 <= r <= 1")
    m = r * sy / sx
    sz = sy * math.sqrt(1 - r * r)
    return RichResult(
        title="Y = mX + Z model from sigma_x, sigma_y and r",
        summary_lines=[("m", m), ("sigma_z", sz)],
        payload={"m": m, "sigma_z": sz},
    )


def cheatsheet():
    return "linmodelinv: m = r sigma_y/sigma_x, sigma_z = sigma_y sqrt(1-r^2). Morin (2016) eq (6.18)."
