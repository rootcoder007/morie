# morie.fn -- function file (rootcoder007/morie)
"""Universal kriging residual"""


def uk_residual(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Universal kriging residual

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.kgunr.uk_residual is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


uk_r = uk_residual


def cheatsheet() -> str:
    return "uk_residual({}) -> Universal kriging residual"


# compact alias per ledger/NAMING.md
ukresidual = uk_residual
