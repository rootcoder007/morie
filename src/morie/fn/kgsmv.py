# morie.fn -- function file (rootcoder007/morie)
"""Simple kriging variance"""


def sk_variance(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Simple kriging variance

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.kgsmv.sk_variance is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


sk_v = sk_variance


def cheatsheet() -> str:
    return "sk_variance({}) -> Simple kriging variance"


# compact alias per ledger/NAMING.md
skvariance = sk_variance
