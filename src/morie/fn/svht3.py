"""Three-candidate spatial equilibrium"""


def hotelling_3cand(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Three-candidate spatial equilibrium

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svht3.hotelling_3cand is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


hote = hotelling_3cand


def cheatsheet() -> str:
    return "hotelling_3cand({}) -> Three-candidate spatial equilibrium"


# compact alias per ledger/NAMING.md
hotelling3cand = hotelling_3cand
