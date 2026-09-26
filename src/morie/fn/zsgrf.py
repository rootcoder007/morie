"""Focal grid statistics"""


def grid_focal(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Focal grid statistics

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zsgrf.grid_focal is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


grid = grid_focal


def cheatsheet() -> str:
    return "grid_focal({}) -> Focal grid statistics"


# compact alias per ledger/NAMING.md
gridfocal = grid_focal
