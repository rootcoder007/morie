# morie.fn -- function file (rootcoder007/morie)
"""INDSCAL subject weights"""


def indscal_weights(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    INDSCAL subject weights

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.msin2.indscal_weights is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


inds = indscal_weights


def cheatsheet() -> str:
    return "indscal_weights({}) -> INDSCAL subject weights"


# compact alias per ledger/NAMING.md
indscalweights = indscal_weights
