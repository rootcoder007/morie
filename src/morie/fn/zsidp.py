"""IDW power parameter optimization"""


def idw_power(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    IDW power parameter optimization

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zsidp.idw_power is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


idw_ = idw_power


def cheatsheet() -> str:
    return "idw_power({}) -> IDW power parameter optimization"


# compact alias per ledger/NAMING.md
idwpower = idw_power
