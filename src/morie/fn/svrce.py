"""Roll call classification error"""


def roll_call_error(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Roll call classification error

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svrce.roll_call_error is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


roll = roll_call_error


def cheatsheet() -> str:
    return "roll_call_error({}) -> Roll call classification error"


# compact alias per ledger/NAMING.md
rollcallerror = roll_call_error
