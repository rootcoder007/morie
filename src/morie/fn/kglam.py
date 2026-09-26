# morie.fn -- function file (rootcoder007/morie)
"""Kriging weights (lambda)"""


def kriging_lambda(values, x, y=None, *, model="spherical"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Kriging weights (lambda)

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.kglam.kriging_lambda is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


krig = kriging_lambda


def cheatsheet() -> str:
    return "kriging_lambda({}) -> Kriging weights (lambda)"


# compact alias per ledger/NAMING.md
kriginglambda = kriging_lambda
