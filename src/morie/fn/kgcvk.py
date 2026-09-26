# morie.fn -- function file (rootcoder007/morie)
"""Kriging k-fold cross-validation"""


def kriging_cv_kfold(values, x, y=None, *, model="spherical"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Kriging k-fold cross-validation

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.kgcvk.kriging_cv_kfold is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


krig = kriging_cv_kfold


def cheatsheet() -> str:
    return "kriging_cv_kfold({}) -> Kriging k-fold cross-validation"


# compact alias per ledger/NAMING.md
krigingcvkfold = kriging_cv_kfold
