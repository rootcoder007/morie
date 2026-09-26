"""Deegan-Packel power index"""


def deegan_packel(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Deegan-Packel power index

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svdgn.deegan_packel is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


deeg = deegan_packel


def cheatsheet() -> str:
    return "deegan_packel({}) -> Deegan-Packel power index"


# compact alias per ledger/NAMING.md
deeganpackel = deegan_packel
