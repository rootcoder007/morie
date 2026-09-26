# morie.fn -- function file (rootcoder007/morie)
"""Nearest-neighbor G-function"""


def g_function(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Nearest-neighbor G-function

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.ptgfn.g_function is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


g_fu = g_function


def cheatsheet() -> str:
    return "g_function({}) -> Nearest-neighbor G-function"


# compact alias per ledger/NAMING.md
gfunction = g_function
