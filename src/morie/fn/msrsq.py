# morie.fn -- function file (rootcoder007/morie)
"""MDS R-squared goodness of fit"""


def mds_rsq(X, *, ndim=2):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    MDS R-squared goodness of fit

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.msrsq.mds_rsq is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


mds_ = mds_rsq


def cheatsheet() -> str:
    return "mds_rsq({}) -> MDS R-squared goodness of fit"


# compact alias per ledger/NAMING.md
mdsrsq = mds_rsq
