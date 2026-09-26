"""Minimum winning coalition size"""


def coalition_size(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Minimum winning coalition size

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svcls.coalition_size is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


coal = coalition_size


def cheatsheet() -> str:
    return "coalition_size({}) -> Minimum winning coalition size"


# compact alias per ledger/NAMING.md
coalitionsize = coalition_size
