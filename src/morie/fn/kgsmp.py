# morie.fn -- function file (rootcoder007/morie)
"""Simple kriging prediction"""


def simple_kriging(values, x, y=None, *, model="spherical"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Simple kriging prediction

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.kgsmp.simple_kriging is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


simp = simple_kriging


def cheatsheet() -> str:
    return "simple_kriging({}) -> Simple kriging prediction"


# compact alias per ledger/NAMING.md
simplekriging = simple_kriging
