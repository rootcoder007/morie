"""Roll call vote probability model"""


def roll_call_prob(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Roll call vote probability model

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svrcp.roll_call_prob is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


roll = roll_call_prob


def cheatsheet() -> str:
    return "roll_call_prob({}) -> Roll call vote probability model"


# compact alias per ledger/NAMING.md
rollcallprob = roll_call_prob
