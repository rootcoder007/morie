# morie.fn -- function file (rootcoder007/morie)
"""W-NOMINATE vote probability"""


def wnominate_prob(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    W-NOMINATE vote probability

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.nmwnp.wnominate_prob is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


wnom = wnominate_prob


def cheatsheet() -> str:
    return "wnominate_prob({}) -> W-NOMINATE vote probability"


# compact alias per ledger/NAMING.md
wnominateprob = wnominate_prob
