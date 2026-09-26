"""GWR bandwidth selection"""


def gwr_bandwidth(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    GWR bandwidth selection

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.xrgwb.gwr_bandwidth is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


gwr_ = gwr_bandwidth


def cheatsheet() -> str:
    return "gwr_bandwidth({}) -> GWR bandwidth selection"


# compact alias per ledger/NAMING.md
gwrbandwidth = gwr_bandwidth
