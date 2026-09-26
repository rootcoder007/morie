# morie.fn -- function file (rootcoder007/morie)
"""Kriging trend surface"""


def kriging_trend_surface(values, x, y=None, *, model="spherical"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Kriging trend surface

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.kgtrn.kriging_trend_surface is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


krig = kriging_trend_surface


def cheatsheet() -> str:
    return "kriging_trend_surface({}) -> Kriging trend surface"
