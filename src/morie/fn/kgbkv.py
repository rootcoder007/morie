# morie.fn -- function file (rootcoder007/morie)
"""Block kriging variance"""


def bk_variance(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Block kriging variance

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.kgbkv.bk_variance is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


bk_v = bk_variance


def cheatsheet() -> str:
    return "bk_variance({}) -> Block kriging variance"


# compact alias per ledger/NAMING.md
bkvariance = bk_variance
