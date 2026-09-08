"""Standard deviation sigma_X = sqrt(Var(X)).

Morin (2016), Probability: For the Enthusiastic Beginner, eq (3.39).
"""


from ._richresult import RichResult

__all__ = ["sdfromvar"]


def sdfromvar(var_x):
    """Standard deviation sigma_X = sqrt(Var(X)).

    Parameters
    ----------
    var_x : float
        A variance, >= 0.

    Returns
    -------
    RichResult
        Keys: variance, sd.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (3.39).
    """
    v = float(var_x)
    if v < 0:
        raise ValueError("variance must be >= 0")
    payload = {"variance": v, "sd": v ** 0.5}
    lines = [("sigma", v ** 0.5)]
    return RichResult(
        title="Standard deviation sigma_X = sqrt(Var(X)).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "sdfromvar: sigma = sqrt(Var(X)). Morin (2016) eq (3.39)."
