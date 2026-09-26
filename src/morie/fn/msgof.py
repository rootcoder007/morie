# morie.fn -- function file (rootcoder007/morie)
"""MDS goodness of fit"""


def mds_gof(X, *, ndim=2):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    MDS goodness of fit

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.msgof.mds_gof is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


mds_ = mds_gof


def cheatsheet() -> str:
    return "mds_gof({}) -> MDS goodness of fit"


# compact alias per ledger/NAMING.md
mdsgof = mds_gof
