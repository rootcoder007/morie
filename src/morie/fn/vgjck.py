"""Variogram jackknife"""


def vario_jackknife(coords, values, *, nbins=15):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Variogram jackknife

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.vgjck.vario_jackknife is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


vari = vario_jackknife


def cheatsheet() -> str:
    return "vario_jackknife({}) -> Variogram jackknife"


# compact alias per ledger/NAMING.md
variojackknife = vario_jackknife
