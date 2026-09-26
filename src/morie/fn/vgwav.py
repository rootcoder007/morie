"""Wave (hole-effect) variogram"""


def vario_wave(coords, values, *, nbins=15):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Wave (hole-effect) variogram

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.vgwav.vario_wave is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


vari = vario_wave


def cheatsheet() -> str:
    return "vario_wave({}) -> Wave (hole-effect) variogram"


# compact alias per ledger/NAMING.md
variowave = vario_wave
