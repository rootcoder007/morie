"""Exponential second moment E(T^2) = 2 tau^2 and Var(T) = tau^2.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (4.85).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["expmom2"]


def expmom2(tau):
    """Exponential second moment E(T^2) = 2 tau^2 and Var(T) = tau^2.

    Parameters
    ----------
    tau : float
        Mean waiting time, > 0.

    Returns
    -------
    RichResult
        Keys: second_moment, variance.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (4.85).
    """
    mean, second, var = _morin.exponential_moments(tau)
    payload = {"second_moment": second, "variance": var}
    lines = [("E(T^2)", second), ("Var(T)", var)]
    return RichResult(
        title="Exponential second moment E(T^2) = 2 tau^2 and Var(T) = tau^2.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "expmom2: E(T^2) = 2 tau^2, Var(T) = tau^2. Morin (2016) eq (4.85)."
