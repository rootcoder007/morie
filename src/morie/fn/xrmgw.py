"""MGWR estimation"""


def mgwr_estimate(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    MGWR estimation

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.xrmgw.mgwr_estimate is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


mgwr = mgwr_estimate


def cheatsheet() -> str:
    return "mgwr_estimate({}) -> MGWR estimation"


# compact alias per ledger/NAMING.md
mgwrestimate = mgwr_estimate
