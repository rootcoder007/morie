"""Congressional party position"""


def party_congress(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Congressional party position

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svppc.party_congress is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


part = party_congress


def cheatsheet() -> str:
    return "party_congress({}) -> Congressional party position"


# compact alias per ledger/NAMING.md
partycongress = party_congress
