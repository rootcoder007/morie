"""2D spatial polarization"""


def polarization_2d(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    2D spatial polarization

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svpl2.polarization_2d is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


pola = polarization_2d


def cheatsheet() -> str:
    return "polarization_2d({}) -> 2D spatial polarization"


# compact alias per ledger/NAMING.md
polarization2d = polarization_2d
