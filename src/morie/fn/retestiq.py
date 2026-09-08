"""IQ test-retest worked example: r = 1/sqrt(2) with sigma_y = 15.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (6.38).
"""

import math

from . import _morin

from ._richresult import RichResult

__all__ = ["retestiq"]


def retestiq():
    """IQ test-retest worked example: r = 1/sqrt(2) with sigma_y = 15.

    Equal signal and noise spreads of 15/sqrt(2) reproduce the
    observed IQ spread of 15 and a retest correlation of about 0.71.

    Returns
    -------
    RichResult
        Keys: r, sigma_y.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (6.38).
    """
    sigma_x = 15.0 / math.sqrt(2.0)
    mu_y, sigma_y, r = _morin.linear_model_stats(1.0, 0.0, sigma_x, 0.0, sigma_x)
    payload = {"r": r, "sigma_y": sigma_y}
    lines = [("r", r), ("sigma_y", sigma_y)]
    return RichResult(
        title="IQ test-retest worked example: r = 1/sqrt(2) with sigma_y = 15.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "retestiq: IQ test-retest r = 1/sqrt(2), sigma_y = 15. Morin (2016) eq (6.38)."
