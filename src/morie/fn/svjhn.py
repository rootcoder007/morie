"""Johnston power index"""


def johnston_power(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Johnston power index

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svjhn.johnston_power is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


john = johnston_power


def cheatsheet() -> str:
    return "johnston_power({}) -> Johnston power index"


# compact alias per ledger/NAMING.md
johnstonpower = johnston_power
