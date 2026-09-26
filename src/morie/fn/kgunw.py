# morie.fn -- function file (rootcoder007/morie)
"""Universal kriging weights"""


def uk_weights(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Universal kriging weights

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.kgunw.uk_weights is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


uk_w = uk_weights


def cheatsheet() -> str:
    return "uk_weights({}) -> Universal kriging weights"


# compact alias per ledger/NAMING.md
ukweights = uk_weights
