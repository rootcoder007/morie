"""Modified Shepard interpolation"""


def idw_shepard(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Modified Shepard interpolation

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zsids.idw_shepard is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


idw_ = idw_shepard


def cheatsheet() -> str:
    return "idw_shepard({}) -> Modified Shepard interpolation"


# compact alias per ledger/NAMING.md
idwshepard = idw_shepard
