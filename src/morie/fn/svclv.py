"""Coalition value in spatial game"""


def coalition_value(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Coalition value in spatial game

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svclv.coalition_value is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


coal = coalition_value


def cheatsheet() -> str:
    return "coalition_value({}) -> Coalition value in spatial game"


# compact alias per ledger/NAMING.md
coalitionvalue = coalition_value
