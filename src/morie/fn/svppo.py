"""Party position estimation"""


def party_position(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Party position estimation

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svppo.party_position is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


part = party_position


def cheatsheet() -> str:
    return "party_position({}) -> Party position estimation"


# compact alias per ledger/NAMING.md
partyposition = party_position
