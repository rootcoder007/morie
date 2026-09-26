"""Spatial Theil decomposition"""


def theil_spatial(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Spatial Theil decomposition

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zethl.theil_spatial is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


thei = theil_spatial


def cheatsheet() -> str:
    return "theil_spatial({}) -> Spatial Theil decomposition"


# compact alias per ledger/NAMING.md
theilspatial = theil_spatial
