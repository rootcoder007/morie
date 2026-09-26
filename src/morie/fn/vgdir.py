"""Directional variogram"""


def directional_vario(coords, values, *, nbins=15):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Directional variogram

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.vgdir.directional_vario is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


dire = directional_vario


def cheatsheet() -> str:
    return "directional_vario({}) -> Directional variogram"
