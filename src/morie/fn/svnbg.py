"""Nash bargaining in spatial game"""


def nash_bargain_sp(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Nash bargaining in spatial game

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svnbg.nash_bargain_sp is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


nash = nash_bargain_sp


def cheatsheet() -> str:
    return "nash_bargain_sp({}) -> Nash bargaining in spatial game"


# compact alias per ledger/NAMING.md
nashbargainsp = nash_bargain_sp
