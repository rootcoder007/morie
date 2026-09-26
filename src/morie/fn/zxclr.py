"""Centered log-ratio spatial"""


def clr_spatial(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Centered log-ratio spatial

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.zxclr.clr_spatial is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


clr_ = clr_spatial


def cheatsheet() -> str:
    return "clr_spatial({}) -> Centered log-ratio spatial"


# compact alias per ledger/NAMING.md
clrspatial = clr_spatial
