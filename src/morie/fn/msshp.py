# morie.fn -- function file (rootcoder007/morie)
"""Shepard diagram values"""


def shepard_diag(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Shepard diagram values

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.msshp.shepard_diag is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


shep = shepard_diag


def cheatsheet() -> str:
    return "shepard_diag({}) -> Shepard diagram values"


# compact alias per ledger/NAMING.md
sheparddiag = shepard_diag
