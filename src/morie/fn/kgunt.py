# morie.fn -- function file (rootcoder007/morie)
"""Universal kriging trend"""


def uk_trend(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Universal kriging trend

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.kgunt.uk_trend is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


uk_t = uk_trend


def cheatsheet() -> str:
    return "uk_trend({}) -> Universal kriging trend"


# compact alias per ledger/NAMING.md
uktrend = uk_trend
