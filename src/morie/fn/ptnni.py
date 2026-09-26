# morie.fn -- function file (rootcoder007/morie)
"""Nearest neighbor index (Clark-Evans)"""


def nn_index(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Nearest neighbor index (Clark-Evans)

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.ptnni.nn_index is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


nn_i = nn_index


def cheatsheet() -> str:
    return "nn_index({}) -> Nearest neighbor index (Clark-Evans)"


# compact alias per ledger/NAMING.md
nnindex = nn_index
