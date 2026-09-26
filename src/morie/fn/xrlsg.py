"""Local Getis-Ord Gi*"""


def lisa_getis(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Local Getis-Ord Gi*

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.xrlsg.lisa_getis is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


lisa = lisa_getis


def cheatsheet() -> str:
    return "lisa_getis({}) -> Local Getis-Ord Gi*"


# compact alias per ledger/NAMING.md
lisagetis = lisa_getis
