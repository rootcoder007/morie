"""Local Moran's I (LISA)"""


def lisa_local(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Local Moran's I (LISA)

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.xrlsa.lisa_local is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


lisa = lisa_local


def cheatsheet() -> str:
    return "lisa_local({}) -> Local Moran's I (LISA)"


# compact alias per ledger/NAMING.md
lisalocal = lisa_local
