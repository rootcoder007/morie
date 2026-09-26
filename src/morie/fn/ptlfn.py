# morie.fn -- function file (rootcoder007/morie)
"""L-function (variance-stabilized K)"""


def l_function(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    L-function (variance-stabilized K)

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.ptlfn.l_function is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


l_fu = l_function


def cheatsheet() -> str:
    return "l_function({}) -> L-function (variance-stabilized K)"


# compact alias per ledger/NAMING.md
lfunction = l_function
