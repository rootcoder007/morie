# morie.fn -- function file (rootcoder007/morie)
"""Universal kriging variance"""


def uk_variance(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Universal kriging variance

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.kgunv.uk_variance is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


uk_v = uk_variance


def cheatsheet() -> str:
    return "uk_variance({}) -> Universal kriging variance"


# compact alias per ledger/NAMING.md
ukvariance = uk_variance
