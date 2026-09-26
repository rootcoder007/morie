"""Contour line generation"""


def contour_lines(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Contour line generation

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zscnt.contour_lines is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


cont = contour_lines


def cheatsheet() -> str:
    return "contour_lines({}) -> Contour line generation"


# compact alias per ledger/NAMING.md
contourlines = contour_lines
