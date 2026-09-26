"""Pseudo-cross-variogram"""


def pseudo_cross_vario(coords, values, *, nbins=15):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Pseudo-cross-variogram

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.vgpxv.pseudo_cross_vario is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


pseu = pseudo_cross_vario


def cheatsheet() -> str:
    return "pseudo_cross_vario({}) -> Pseudo-cross-variogram"
