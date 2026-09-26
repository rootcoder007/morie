# morie.fn -- function file (rootcoder007/morie)
"""Party divergence measure"""


def party_diverge(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Party divergence measure

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.nmdmp.party_diverge is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


part = party_diverge


def cheatsheet() -> str:
    return "party_diverge({}) -> Party divergence measure"


# compact alias per ledger/NAMING.md
partydiverge = party_diverge
