"""Test-retest correlation from a signal spread and a noise spread.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (6.37).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["retestr"]


def retestr(sigma_signal, sigma_noise):
    """Test-retest correlation from a signal spread and a noise spread.

    r = sigma_signal / sqrt(sigma_signal^2 + sigma_noise^2); equal
    signal and noise give r = 1/sqrt(2).

    Parameters
    ----------
    sigma_signal : float
        Spread of the underlying ability, >= 0.
    sigma_noise : float
        Spread of the measurement noise, >= 0.

    Returns
    -------
    RichResult
        Keys: r.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (6.37).
    """
    mu_y, sigma_y, r = _morin.linear_model_stats(
        1.0, 0.0, sigma_signal, 0.0, sigma_noise)
    payload = {"r": r}
    lines = [("r", r)]
    return RichResult(
        title="Test-retest correlation from a signal spread and a noise spread.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "retestr: Test-retest r from signal and noise spreads. Morin (2016) eq (6.37)."
