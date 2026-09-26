"""Optimal cutting line"""


def cut_line(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Optimal cutting line

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svcli.cut_line is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


cut_ = cut_line


def cheatsheet() -> str:
    return "cut_line({}) -> Optimal cutting line"


# compact alias per ledger/NAMING.md
cutline = cut_line
