# morie.fn -- function file (rootcoder007/morie)
"""Indicator kriging probability"""


def ik_probability(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Indicator kriging probability

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.kgikp.ik_probability is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


ik_p = ik_probability


def cheatsheet() -> str:
    return "ik_probability({}) -> Indicator kriging probability"


# compact alias per ledger/NAMING.md
ikprobability = ik_probability
