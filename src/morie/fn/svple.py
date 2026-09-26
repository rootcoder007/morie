"""Esteban-Ray polarization index"""


def polarization_er(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Esteban-Ray polarization index

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svple.polarization_er is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


pola = polarization_er


def cheatsheet() -> str:
    return "polarization_er({}) -> Esteban-Ray polarization index"


# compact alias per ledger/NAMING.md
polarizationer = polarization_er
