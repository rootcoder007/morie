# morie.fn -- function file (rootcoder007/morie)
"""Cox (doubly stochastic) process"""


def cox_process(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Cox (doubly stochastic) process

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.ptcox.cox_process is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


cox_ = cox_process


def cheatsheet() -> str:
    return "cox_process({}) -> Cox (doubly stochastic) process"


# compact alias per ledger/NAMING.md
coxprocess = cox_process
