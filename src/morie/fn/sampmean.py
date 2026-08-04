"""Sample mean X-bar = (X1 + ... + Xn)/n.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (3.54).
"""

from . import _array_core as np

from . import _morin

from ._richresult import RichResult

__all__ = ["sampmean"]


def sampmean(x):
    """Sample mean X-bar = (X1 + ... + Xn)/n.

    Parameters
    ----------
    x : array-like
        Numeric sample, non-empty.

    Returns
    -------
    RichResult
        Keys: mean, n.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (3.54).
    """
    value = _morin.sample_mean(x)
    payload = {"mean": value, "n": int(np.atleast_1d(x).size)}
    lines = [("X-bar", value)]
    return RichResult(
        title="Sample mean X-bar = (X1 + ... + Xn)/n.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "sampmean: Sample mean (X1 + ... + Xn)/n. Morin (2016) eq (3.54)."
