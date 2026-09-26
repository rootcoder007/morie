"""Circular variogram model"""


def vario_circular(coords, values, *, nbins=15):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Circular variogram model

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.vgcir.vario_circular is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


vari = vario_circular


def cheatsheet() -> str:
    return "vario_circular({}) -> Circular variogram model"


# compact alias per ledger/NAMING.md
variocircular = vario_circular
