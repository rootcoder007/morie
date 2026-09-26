"""Variogram WLS fitting"""


def vario_fit_wls(coords, values, *, nbins=15):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Variogram WLS fitting

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.vgfit.vario_fit_wls is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


vari = vario_fit_wls


def cheatsheet() -> str:
    return "vario_fit_wls({}) -> Variogram WLS fitting"


# compact alias per ledger/NAMING.md
variofitwls = vario_fit_wls
