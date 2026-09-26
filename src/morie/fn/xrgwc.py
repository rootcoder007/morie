"""GWR local coefficients"""


def gwr_coefficients(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    GWR local coefficients

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.xrgwc.gwr_coefficients is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


gwr_ = gwr_coefficients


def cheatsheet() -> str:
    return "gwr_coefficients({}) -> GWR local coefficients"
