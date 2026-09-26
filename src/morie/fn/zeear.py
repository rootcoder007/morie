"""Ecological regression (Poisson)"""


def ecological_reg(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Ecological regression (Poisson)

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zeear.ecological_reg is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


ecol = ecological_reg


def cheatsheet() -> str:
    return "ecological_reg({}) -> Ecological regression (Poisson)"


# compact alias per ledger/NAMING.md
ecologicalreg = ecological_reg
