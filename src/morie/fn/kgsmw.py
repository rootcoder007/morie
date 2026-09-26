# morie.fn -- function file (rootcoder007/morie)
"""Simple kriging weights"""


def sk_weights(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Simple kriging weights

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.kgsmw.sk_weights is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


sk_w = sk_weights


def cheatsheet() -> str:
    return "sk_weights({}) -> Simple kriging weights"


# compact alias per ledger/NAMING.md
skweights = sk_weights
