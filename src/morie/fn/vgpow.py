"""Power variogram model"""


def vario_power(coords, values, *, nbins=15):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Power variogram model

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.vgpow.vario_power is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


vari = vario_power


def cheatsheet() -> str:
    return "vario_power({}) -> Power variogram model"


# compact alias per ledger/NAMING.md
variopower = vario_power
