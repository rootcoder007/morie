# morie.fn -- function file (rootcoder007/morie)
"""Kriging residual map"""


def kriging_residual_map(values, x, y=None, *, model="spherical"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Kriging residual map

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.kgres.kriging_residual_map is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


krig = kriging_residual_map


def cheatsheet() -> str:
    return "kriging_residual_map({}) -> Kriging residual map"
