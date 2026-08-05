"""Mean of Y = mX + Z: mu_y = m mu_x + mu_z.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (6.4).
"""


from ._richresult import RichResult

__all__ = ["muy"]


def muy(m, mu_x, mu_z):
    """Mean of Y = mX + Z: mu_y = m mu_x + mu_z.

    Parameters
    ----------
    m : float
        Slope.
    mu_x : float
        Mean of the signal.
    mu_z : float
        Mean of the noise.

    Returns
    -------
    RichResult
        Keys: mu_y.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (6.4).
    """
    mu_y = float(m) * float(mu_x) + float(mu_z)
    payload = {"mu_y": mu_y}
    lines = [("mu_y", mu_y)]
    return RichResult(
        title="Mean of Y = mX + Z: mu_y = m mu_x + mu_z.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "muy: mu_y = m mu_x + mu_z. Morin (2016) eq (6.4)."
