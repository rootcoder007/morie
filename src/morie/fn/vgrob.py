"""Robust semivariogram (Cressie-Hawkins)"""


def vario_robust(coords, values, *, nbins=15):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Robust semivariogram (Cressie-Hawkins)

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.vgrob.vario_robust is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


vari = vario_robust


def cheatsheet() -> str:
    return "vario_robust({}) -> Robust semivariogram (Cressie-Hawkins)"


# compact alias per ledger/NAMING.md
variorobust = vario_robust
