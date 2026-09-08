"""The constant predictor minimizing squared error is the mean.

Morin (2016), Probability: For the Enthusiastic Beginner, eqs (6.22)-(6.23).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["bestconst"]


def bestconst(y):
    """The constant predictor minimizing squared error is the mean.

    Two probe constants either side of the mean are evaluated; if
    either beats the mean the routine raises rather than returning a
    wrong optimum.

    Parameters
    ----------
    y : array-like
        Outcomes, non-empty.

    Returns
    -------
    RichResult
        Keys: best_prediction, mse.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eqs (6.22)-(6.23).
    """
    mu, mse = _morin.best_constant_predictor(y)
    payload = {"best_prediction": mu, "mse": mse}
    lines = [("y_p = mean", mu), ("mse", mse)]
    return RichResult(
        title="The constant predictor minimizing squared error is the mean.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "bestconst: The best constant predictor is the mean. Morin (2016) eqs (6.22)-(6.23)."
