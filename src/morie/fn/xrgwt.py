"""GWR local t-values"""


def gwr_tvalues(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    GWR local t-values

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.xrgwt.gwr_tvalues is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


gwr_ = gwr_tvalues


def cheatsheet() -> str:
    return "gwr_tvalues({}) -> GWR local t-values"


# compact alias per ledger/NAMING.md
gwrtvalues = gwr_tvalues
