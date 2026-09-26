# morie.fn -- function file (rootcoder007/morie)
"""Kriging LOO cross-validation"""


def kriging_cv_loo(values, x, y=None, *, model="spherical"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Kriging LOO cross-validation

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.kgcvl.kriging_cv_loo is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


krig = kriging_cv_loo


def cheatsheet() -> str:
    return "kriging_cv_loo({}) -> Kriging LOO cross-validation"


# compact alias per ledger/NAMING.md
krigingcvloo = kriging_cv_loo
