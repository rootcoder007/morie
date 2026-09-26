# morie.fn -- function file (rootcoder007/morie)
"""Kriging right-hand side vector"""


def kriging_rhs(values, x, y=None, *, model="spherical"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Kriging right-hand side vector

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.kgrhs.kriging_rhs is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


krig = kriging_rhs


def cheatsheet() -> str:
    return "kriging_rhs({}) -> Kriging right-hand side vector"


# compact alias per ledger/NAMING.md
krigingrhs = kriging_rhs
