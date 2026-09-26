# morie.fn -- function file (rootcoder007/morie)
"""Classical MDS (Torgerson)"""


def classical_mds(X, *, ndim=2):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Classical MDS (Torgerson)

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.mscls.classical_mds is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


clas = classical_mds


def cheatsheet() -> str:
    return "classical_mds({}) -> Classical MDS (Torgerson)"


# compact alias per ledger/NAMING.md
classicalmds = classical_mds
