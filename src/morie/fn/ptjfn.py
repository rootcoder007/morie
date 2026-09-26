# morie.fn -- function file (rootcoder007/morie)
"""J-function (ratio F/G)"""


def j_function(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    J-function (ratio F/G)

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.ptjfn.j_function is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


j_fu = j_function


def cheatsheet() -> str:
    return "j_function({}) -> J-function (ratio F/G)"


# compact alias per ledger/NAMING.md
jfunction = j_function
