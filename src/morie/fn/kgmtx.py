# morie.fn -- function file (rootcoder007/morie)
"""Kriging system matrix"""


def kriging_matrix(values, x, y=None, *, model="spherical"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Kriging system matrix

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.kgmtx.kriging_matrix is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


krig = kriging_matrix


def cheatsheet() -> str:
    return "kriging_matrix({}) -> Kriging system matrix"


# compact alias per ledger/NAMING.md
krigingmatrix = kriging_matrix
