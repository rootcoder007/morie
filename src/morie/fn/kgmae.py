# morie.fn -- function file (rootcoder007/morie)
"""Kriging MAE"""


def kriging_mae(values, x, y=None, *, model="spherical"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Kriging MAE

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.kgmae.kriging_mae is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


krig = kriging_mae


def cheatsheet() -> str:
    return "kriging_mae({}) -> Kriging MAE"


# compact alias per ledger/NAMING.md
krigingmae = kriging_mae
