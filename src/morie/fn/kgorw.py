# morie.fn -- function file (rootcoder007/morie)
"""Ordinary kriging weights"""


def ok_weights(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Ordinary kriging weights

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.kgorw.ok_weights is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


ok_w = ok_weights


def cheatsheet() -> str:
    return "ok_weights({}) -> Ordinary kriging weights"


# compact alias per ledger/NAMING.md
okweights = ok_weights
