"""Variogram REML fitting"""


def vario_fit_reml(coords, values, *, nbins=15):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Variogram REML fitting

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.vgfrl.vario_fit_reml is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


vari = vario_fit_reml


def cheatsheet() -> str:
    return "vario_fit_reml({}) -> Variogram REML fitting"


# compact alias per ledger/NAMING.md
variofitreml = vario_fit_reml
