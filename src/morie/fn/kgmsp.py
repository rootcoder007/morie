# morie.fn -- function file (rootcoder007/morie)
"""Kriging MSPE"""


def kriging_mspe(values, x, y=None, *, model="spherical"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Kriging MSPE

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.kgmsp.kriging_mspe is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


krig = kriging_mspe


def cheatsheet() -> str:
    return "kriging_mspe({}) -> Kriging MSPE"


# compact alias per ledger/NAMING.md
krigingmspe = kriging_mspe
