# morie.fn -- function file (rootcoder007/morie)
"""Block kriging discretization"""


def bk_discretize(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Block kriging discretization

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.kgbkd.bk_discretize is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


bk_d = bk_discretize


def cheatsheet() -> str:
    return "bk_discretize({}) -> Block kriging discretization"


# compact alias per ledger/NAMING.md
bkdiscretize = bk_discretize
