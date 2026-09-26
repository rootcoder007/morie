"""Correlogram function"""


def correlogram(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Correlogram function

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.vgcrf.correlogram is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


corr = correlogram


def cheatsheet() -> str:
    return "correlogram({}) -> Correlogram function"
