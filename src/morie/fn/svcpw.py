"""Copeland spatial winner"""


def copeland_winner(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Copeland spatial winner

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svcpw.copeland_winner is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


cope = copeland_winner


def cheatsheet() -> str:
    return "copeland_winner({}) -> Copeland spatial winner"


# compact alias per ledger/NAMING.md
copelandwinner = copeland_winner
