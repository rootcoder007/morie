# morie.fn -- function file (rootcoder007/morie)
"""Ordinary kriging prediction"""


def ordinary_kriging(values, x, y=None, *, model="spherical"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Ordinary kriging prediction

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.kgord.ordinary_kriging is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


ordi = ordinary_kriging


def cheatsheet() -> str:
    return "ordinary_kriging({}) -> Ordinary kriging prediction"
