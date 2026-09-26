"""Zonal grid statistics"""


def grid_zonal(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Zonal grid statistics

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zsgrz.grid_zonal is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


grid = grid_zonal


def cheatsheet() -> str:
    return "grid_zonal({}) -> Zonal grid statistics"


# compact alias per ledger/NAMING.md
gridzonal = grid_zonal
