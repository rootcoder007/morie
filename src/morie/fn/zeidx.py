"""IDW exposure interpolation"""


def idw_exposure(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    IDW exposure interpolation

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zeidx.idw_exposure is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


idw_ = idw_exposure


def cheatsheet() -> str:
    return "idw_exposure({}) -> IDW exposure interpolation"


# compact alias per ledger/NAMING.md
idwexposure = idw_exposure
