"""Betti numbers computation"""


def betti_numbers(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Betti numbers computation

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.zxbet.betti_numbers is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


bett = betti_numbers


def cheatsheet() -> str:
    return "betti_numbers({}) -> Betti numbers computation"


# compact alias per ledger/NAMING.md
bettinumbers = betti_numbers
