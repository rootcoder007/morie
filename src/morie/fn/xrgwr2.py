"""GWR local R-squared"""


def gwr_rsquared(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    GWR local R-squared

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.xrgwr2.gwr_rsquared is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


gwr_ = gwr_rsquared


def cheatsheet() -> str:
    return "gwr_rsquared({}) -> GWR local R-squared"


# compact alias per ledger/NAMING.md
gwrrsquared = gwr_rsquared
