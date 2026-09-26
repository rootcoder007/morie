"""Party manifesto scaling (Wordscores)"""


def party_manifesto(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Party manifesto scaling (Wordscores)

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svppm.party_manifesto is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


part = party_manifesto


def cheatsheet() -> str:
    return "party_manifesto({}) -> Party manifesto scaling (Wordscores)"


# compact alias per ledger/NAMING.md
partymanifesto = party_manifesto
