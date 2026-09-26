"""GWR basic estimation"""


def gwr_basic(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    GWR basic estimation

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.xrgwr.gwr_basic is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


gwr_ = gwr_basic


def cheatsheet() -> str:
    return "gwr_basic({}) -> GWR basic estimation"


# compact alias per ledger/NAMING.md
gwrbasic = gwr_basic
