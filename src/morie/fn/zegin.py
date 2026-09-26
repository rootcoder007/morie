"""Spatial Gini coefficient"""


def gini_spatial(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Spatial Gini coefficient

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zegin.gini_spatial is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


gini = gini_spatial


def cheatsheet() -> str:
    return "gini_spatial({}) -> Spatial Gini coefficient"


# compact alias per ledger/NAMING.md
ginispatial = gini_spatial
