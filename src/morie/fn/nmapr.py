# morie.fn -- function file (rootcoder007/morie)
"""Aggregate PRE"""


def apre_stat(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Aggregate PRE

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.nmapr.apre_stat is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


apre = apre_stat


def cheatsheet() -> str:
    return "apre_stat({}) -> Aggregate PRE"


# compact alias per ledger/NAMING.md
aprestat = apre_stat
