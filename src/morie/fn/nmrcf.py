# morie.fn -- function file (rootcoder007/morie)
"""Roll call filtering"""


def roll_call_filter(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Roll call filtering

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.nmrcf.roll_call_filter is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


roll = roll_call_filter


def cheatsheet() -> str:
    return "roll_call_filter({}) -> Roll call filtering"


# compact alias per ledger/NAMING.md
rollcallfilter = roll_call_filter
