# morie.fn -- function file (rootcoder007/morie)
"""Kriging equivalence to GLS"""


def kriging_equiv(values, x, y=None, *, model="spherical"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Kriging equivalence to GLS

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.kgeqv.kriging_equiv is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


krig = kriging_equiv


def cheatsheet() -> str:
    return "kriging_equiv({}) -> Kriging equivalence to GLS"


# compact alias per ledger/NAMING.md
krigingequiv = kriging_equiv
