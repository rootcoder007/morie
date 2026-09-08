"""Group-average relation: yavg = m xavg.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (6.39).
"""


from ._richresult import RichResult

__all__ = ["groupavg"]


def groupavg(m, xavg):
    """Group-average relation: yavg = m xavg.

    Parameters
    ----------
    m : float
        Slope.
    xavg : float
        Group average of X.

    Returns
    -------
    RichResult
        Keys: yavg.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (6.39).
    """
    value = float(m) * float(xavg)
    payload = {"yavg": value}
    lines = [("yavg", value)]
    return RichResult(
        title="Group-average relation: yavg = m xavg.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "groupavg: yavg = m xavg. Morin (2016) eq (6.39)."
