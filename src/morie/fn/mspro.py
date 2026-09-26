# morie.fn -- function file (rootcoder007/morie)
"""Orthogonal Procrustes rotation"""


def procrustes_orth(X, *, ndim=2):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Orthogonal Procrustes rotation

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.mspro.procrustes_orth is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


proc = procrustes_orth


def cheatsheet() -> str:
    return "procrustes_orth({}) -> Orthogonal Procrustes rotation"


# compact alias per ledger/NAMING.md
procrustesorth = procrustes_orth
