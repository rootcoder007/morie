# morie.fn -- function file (rootcoder007/morie)
"""Shepard residuals"""


def shepard_resid(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Shepard residuals

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.msshr.shepard_resid is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


shep = shepard_resid


def cheatsheet() -> str:
    return "shepard_resid({}) -> Shepard residuals"


# compact alias per ledger/NAMING.md
shepardresid = shepard_resid
