# morie.fn -- function file (rootcoder007/morie)
"""Universal kriging prediction"""


def universal_kriging(values, x, y=None, *, model="spherical"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Universal kriging prediction

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.kguni.universal_kriging is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


univ = universal_kriging


def cheatsheet() -> str:
    return "universal_kriging({}) -> Universal kriging prediction"
