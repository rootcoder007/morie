"""Moran's I on regression residuals"""


def moran_resid(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Moran's I on regression residuals

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.xrmri.moran_resid is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


mora = moran_resid


def cheatsheet() -> str:
    return "moran_resid({}) -> Moran's I on regression residuals"


# compact alias per ledger/NAMING.md
moranresid = moran_resid
