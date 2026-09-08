"""Second-order approximation (1+a)^n ~ e^(na) e^(-na^2/2), valid when n a^3 << 1.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (7.24).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["powexpap2"]


def powexpap2(a, n):
    """Second-order approximation (1+a)^n ~ e^(na) e^(-na^2/2), valid when n a^3 << 1.

    ``valid`` is the decision n a^3 < 0.1.

    Parameters
    ----------
    a : float
        Perturbation, > -1.
    n : float
        Exponent.

    Returns
    -------
    RichResult
        Keys: exact, approx, na3, valid.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (7.24).
    """
    exact, approx, validity = _morin.one_plus_a_to_n(a, n, order=2)
    payload = {
        "exact": exact,
        "approx": approx,
        "na3": validity,
        "valid": validity < 0.1,
    }
    lines = [("approx", approx), ("na^3", validity)]
    return RichResult(
        title="Second-order approximation (1+a)^n ~ e^(na) e^(-na^2/2), valid when n a^3 << 1.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "powexpap2: (1+a)^n ~ e^(na) e^(-na^2/2) to second order. Morin (2016) eq (7.24)."
