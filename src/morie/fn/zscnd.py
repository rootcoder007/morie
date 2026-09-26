"""Conditional simulation"""


def conditional_sim(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Conditional simulation

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zscnd.conditional_sim is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


cond = conditional_sim


def cheatsheet() -> str:
    return "conditional_sim({}) -> Conditional simulation"


# compact alias per ledger/NAMING.md
conditionalsim = conditional_sim
