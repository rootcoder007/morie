"""Roll call simulation"""


def roll_call_sim(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Roll call simulation

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svrcs.roll_call_sim is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


roll = roll_call_sim


def cheatsheet() -> str:
    return "roll_call_sim({}) -> Roll call simulation"


# compact alias per ledger/NAMING.md
rollcallsim = roll_call_sim
