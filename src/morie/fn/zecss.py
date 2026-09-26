"""Spatial CUSUM aberration detection"""


def cusum_spatial(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Spatial CUSUM aberration detection

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zecss.cusum_spatial is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


cusu = cusum_spatial


def cheatsheet() -> str:
    return "cusum_spatial({}) -> Spatial CUSUM aberration detection"


# compact alias per ledger/NAMING.md
cusumspatial = cusum_spatial
