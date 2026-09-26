# morie.fn -- function file (rootcoder007/morie)
"""Nonmetric MDS 2D"""


def nonmetric_2d(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Nonmetric MDS 2D

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.msnm2.nonmetric_2d is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


nonm = nonmetric_2d


def cheatsheet() -> str:
    return "nonmetric_2d({}) -> Nonmetric MDS 2D"


# compact alias per ledger/NAMING.md
nonmetric2d = nonmetric_2d
