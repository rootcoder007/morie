"""Variogram cloud"""


def vario_cloud(coords, values, *, nbins=15):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Variogram cloud

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.vgcld.vario_cloud is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


vari = vario_cloud


def cheatsheet() -> str:
    return "vario_cloud({}) -> Variogram cloud"


# compact alias per ledger/NAMING.md
variocloud = vario_cloud
