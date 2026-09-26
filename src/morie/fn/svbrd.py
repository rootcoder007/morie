"""Borda count in spatial model"""


def borda_spatial(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Borda count in spatial model

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svbrd.borda_spatial is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


bord = borda_spatial


def cheatsheet() -> str:
    return "borda_spatial({}) -> Borda count in spatial model"


# compact alias per ledger/NAMING.md
bordaspatial = borda_spatial
