"""Getis spatial filtering"""


def getis_filter(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Getis spatial filtering

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.xrfgt.getis_filter is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


geti = getis_filter


def cheatsheet() -> str:
    return "getis_filter({}) -> Getis spatial filtering"


# compact alias per ledger/NAMING.md
getisfilter = getis_filter
