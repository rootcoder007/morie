"""Multi-party 2D equilibrium"""


def multiparty_2d(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Multi-party 2D equilibrium

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svmp2.multiparty_2d is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


mult = multiparty_2d


def cheatsheet() -> str:
    return "multiparty_2d({}) -> Multi-party 2D equilibrium"


# compact alias per ledger/NAMING.md
multiparty2d = multiparty_2d
