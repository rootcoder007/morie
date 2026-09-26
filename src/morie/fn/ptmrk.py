# morie.fn -- function file (rootcoder007/morie)
"""Marked point pattern analysis"""


def marked_pp(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Marked point pattern analysis

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.ptmrk.marked_pp is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


mark = marked_pp


def cheatsheet() -> str:
    return "marked_pp({}) -> Marked point pattern analysis"


# compact alias per ledger/NAMING.md
markedpp = marked_pp
