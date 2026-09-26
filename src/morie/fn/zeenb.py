"""Ecological NB regression"""


def ecological_nb(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Ecological NB regression

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zeenb.ecological_nb is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


ecol = ecological_nb


def cheatsheet() -> str:
    return "ecological_nb({}) -> Ecological NB regression"


# compact alias per ledger/NAMING.md
ecologicalnb = ecological_nb
