"""Geographically weighted summary stats"""


def gw_summary(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Geographically weighted summary stats

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.zxgws.gw_summary is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


gw_s = gw_summary


def cheatsheet() -> str:
    return "gw_summary({}) -> Geographically weighted summary stats"


# compact alias per ledger/NAMING.md
gwsummary = gw_summary
