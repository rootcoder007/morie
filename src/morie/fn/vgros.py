"""Variogram rose diagram"""


def vario_rose(coords, values, *, nbins=15):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Variogram rose diagram

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.vgros.vario_rose is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


vari = vario_rose


def cheatsheet() -> str:
    return "vario_rose({}) -> Variogram rose diagram"


# compact alias per ledger/NAMING.md
variorose = vario_rose
