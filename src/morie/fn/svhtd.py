"""Hotelling-Downs convergence equilibrium"""


def hotelling_downs(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Hotelling-Downs convergence equilibrium

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svhtd.hotelling_downs is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


hote = hotelling_downs


def cheatsheet() -> str:
    return "hotelling_downs({}) -> Hotelling-Downs convergence equilibrium"


# compact alias per ledger/NAMING.md
hotellingdowns = hotelling_downs
