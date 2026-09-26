"""Roll call logit model"""


def roll_call_logit(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Roll call logit model

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svrcl.roll_call_logit is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


roll = roll_call_logit


def cheatsheet() -> str:
    return "roll_call_logit({}) -> Roll call logit model"


# compact alias per ledger/NAMING.md
rollcalllogit = roll_call_logit
