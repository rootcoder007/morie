"""Filled contour generation"""


def contour_fill(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Filled contour generation

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zscnf.contour_fill is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


cont = contour_fill


def cheatsheet() -> str:
    return "contour_fill({}) -> Filled contour generation"


# compact alias per ledger/NAMING.md
contourfill = contour_fill
