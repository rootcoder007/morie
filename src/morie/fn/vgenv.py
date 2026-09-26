"""Variogram Monte Carlo envelope"""


def vario_envelope(coords, values, *, nbins=15):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Variogram Monte Carlo envelope

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.vgenv.vario_envelope is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


vari = vario_envelope


def cheatsheet() -> str:
    return "vario_envelope({}) -> Variogram Monte Carlo envelope"


# compact alias per ledger/NAMING.md
varioenvelope = vario_envelope
