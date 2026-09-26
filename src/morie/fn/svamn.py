"""Sequential amendment procedure"""


def amendment_seq(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Sequential amendment procedure

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svamn.amendment_seq is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


amen = amendment_seq


def cheatsheet() -> str:
    return "amendment_seq({}) -> Sequential amendment procedure"


# compact alias per ledger/NAMING.md
amendmentseq = amendment_seq
