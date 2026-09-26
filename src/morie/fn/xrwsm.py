"""Weights symmetrization"""


def w_symmetrize(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Weights symmetrization

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.xrwsm.w_symmetrize is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


w_sy = w_symmetrize


def cheatsheet() -> str:
    return "w_symmetrize({}) -> Weights symmetrization"


# compact alias per ledger/NAMING.md
wsymmetrize = w_symmetrize
