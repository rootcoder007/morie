"""Spatial von Mises distribution"""


def von_mises_sp(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Spatial von Mises distribution

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.zxvms.von_mises_sp is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


von_ = von_mises_sp


def cheatsheet() -> str:
    return "von_mises_sp({}) -> Spatial von Mises distribution"


# compact alias per ledger/NAMING.md
vonmisessp = von_mises_sp
