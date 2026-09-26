"""Cubic variogram model"""


def vario_cubic(coords, values, *, nbins=15):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Cubic variogram model

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.vgcub.vario_cubic is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


vari = vario_cubic


def cheatsheet() -> str:
    return "vario_cubic({}) -> Cubic variogram model"


# compact alias per ledger/NAMING.md
variocubic = vario_cubic
