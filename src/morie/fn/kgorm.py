# morie.fn -- function file (rootcoder007/morie)
"""Ordinary kriging matrix system"""


def ok_matrix(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Ordinary kriging matrix system

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.kgorm.ok_matrix is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


ok_m = ok_matrix


def cheatsheet() -> str:
    return "ok_matrix({}) -> Ordinary kriging matrix system"


# compact alias per ledger/NAMING.md
okmatrix = ok_matrix
