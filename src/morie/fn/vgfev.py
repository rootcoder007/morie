"""Variogram fit evaluation"""


def vario_fit_eval(coords, values, *, nbins=15):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Variogram fit evaluation

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.vgfev.vario_fit_eval is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


vari = vario_fit_eval


def cheatsheet() -> str:
    return "vario_fit_eval({}) -> Variogram fit evaluation"


# compact alias per ledger/NAMING.md
variofiteval = vario_fit_eval
