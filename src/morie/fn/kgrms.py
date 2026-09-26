# morie.fn -- function file (rootcoder007/morie)
"""Kriging RMSE"""


def kriging_rmse(values, x, y=None, *, model="spherical"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Kriging RMSE

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.kgrms.kriging_rmse is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


krig = kriging_rmse


def cheatsheet() -> str:
    return "kriging_rmse({}) -> Kriging RMSE"


# compact alias per ledger/NAMING.md
krigingrmse = kriging_rmse
