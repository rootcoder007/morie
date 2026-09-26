"""IDW interpolation"""


def idw_interp(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    IDW interpolation

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zsidw.idw_interp is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


idw_ = idw_interp


def cheatsheet() -> str:
    return "idw_interp({}) -> IDW interpolation"


# compact alias per ledger/NAMING.md
idwinterp = idw_interp
