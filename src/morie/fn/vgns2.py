"""Nested variogram fitting"""


def vario_nested_fit(coords, values, *, nbins=15):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Nested variogram fitting

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.vgns2.vario_nested_fit is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


vari = vario_nested_fit


def cheatsheet() -> str:
    return "vario_nested_fit({}) -> Nested variogram fitting"


# compact alias per ledger/NAMING.md
varionestedfit = vario_nested_fit
