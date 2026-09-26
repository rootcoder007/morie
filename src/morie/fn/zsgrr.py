"""Grid resampling"""


def grid_resample(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Grid resampling

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zsgrr.grid_resample is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


grid = grid_resample


def cheatsheet() -> str:
    return "grid_resample({}) -> Grid resampling"


# compact alias per ledger/NAMING.md
gridresample = grid_resample
