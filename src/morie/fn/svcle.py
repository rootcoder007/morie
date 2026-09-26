"""Coalition equilibrium (Schofield)"""


def coalition_equil(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Coalition equilibrium (Schofield)

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svcle.coalition_equil is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


coal = coalition_equil


def cheatsheet() -> str:
    return "coalition_equil({}) -> Coalition equilibrium (Schofield)"


# compact alias per ledger/NAMING.md
coalitionequil = coalition_equil
