"""The Y = mX + Z model: mu_y, sigma_y and the correlation r.

Morin (2016), Probability: For the Enthusiastic Beginner, eqs (6.3)-(6.6), (6.17), (6.76).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["linmodel"]


def linmodel(m=1.0, sigma_x=7.5, sigma_z=10.6):
    """The Y = mX + Z model: mu_y, sigma_y and the correlation r.

    mu_y = m mu_x + mu_z with mu_x = mu_z = 0, sigma_y =
    sqrt(m^2 sigma_x^2 + sigma_z^2) and r = m sigma_x / sigma_y.  The
    defaults are the book's worked spread, sigma_y = 13.

    Parameters
    ----------
    m : float
        Slope of the underlying relation.
    sigma_x : float
        Spread of the signal, >= 0.
    sigma_z : float
        Spread of the independent noise, >= 0.

    Returns
    -------
    RichResult
        Keys: mu_y, sigma_y, r, mu_z, sigma_z.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eqs (6.3)-(6.6), (6.17), (6.76).
    """
    mu_y, sigma_y, r = _morin.linear_model_stats(m, 0.0, sigma_x, 0.0, sigma_z)
    payload = {
        "mu_y": mu_y,
        "sigma_y": sigma_y,
        "r": r,
        "mu_z": 0.0,
        "sigma_z": float(sigma_z),
    }
    lines = [("sigma_y", sigma_y), ("r", r)]
    return RichResult(
        title="The Y = mX + Z model: mu_y, sigma_y and the correlation r.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "linmodel: sigma_y and r for the model Y = mX + Z. Morin (2016) eqs (6.3)-(6.6), (6.17), (6.76)."
