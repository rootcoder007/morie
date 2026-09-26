"""Multi-party Nash equilibrium"""


def multiparty_nash(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Multi-party Nash equilibrium

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svmpn.multiparty_nash is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


mult = multiparty_nash


def cheatsheet() -> str:
    return "multiparty_nash({}) -> Multi-party Nash equilibrium"


# compact alias per ledger/NAMING.md
multipartynash = multiparty_nash
