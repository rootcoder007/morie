"""Persistent homology spatial"""


def tda_persistent(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Persistent homology spatial

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.zxtda.tda_persistent is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


tda_ = tda_persistent


def cheatsheet() -> str:
    return "tda_persistent({}) -> Persistent homology spatial"


# compact alias per ledger/NAMING.md
tdapersistent = tda_persistent
