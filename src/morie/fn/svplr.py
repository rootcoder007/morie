"""1D ideological polarization index"""


def polarization_1d(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    1D ideological polarization index

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svplr.polarization_1d is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


pola = polarization_1d


def cheatsheet() -> str:
    return "polarization_1d({}) -> 1D ideological polarization index"


# compact alias per ledger/NAMING.md
polarization1d = polarization_1d
