"""sigma of the number of Heads in n fair flips: sqrt(n)/2.

Morin (2016), Probability: For the Enthusiastic Beginner, eqs (3.48), (3.51).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["sdcoinsum"]


def sdcoinsum(n):
    """sigma of the number of Heads in n fair flips: sqrt(n)/2.

    Parameters
    ----------
    n : int
        Number of fair flips, >= 0.

    Returns
    -------
    RichResult
        Keys: n, sd, sd_tot (an alias of sd kept for eq (3.51)).

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eqs (3.48), (3.51).
    """
    value = _morin.sd_fair_coin_sum(n)
    payload = {"n": int(n), "sd": value, "sd_tot": value}
    lines = [("sqrt(n)/2", value)]
    return RichResult(
        title="sigma of the number of Heads in n fair flips: sqrt(n)/2.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "sdcoinsum: sigma of the Heads count in n fair flips = sqrt(n)/2. Morin (2016) eqs (3.48), (3.51)."
