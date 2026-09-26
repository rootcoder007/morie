# morie.fn -- function file (rootcoder007/morie)
"""Ordinary kriging variance"""


def ok_variance(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Ordinary kriging variance

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.kgorv.ok_variance is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


ok_v = ok_variance


def cheatsheet() -> str:
    return "ok_variance({}) -> Ordinary kriging variance"


# compact alias per ledger/NAMING.md
okvariance = ok_variance
