# morie.fn -- function file (rootcoder007/morie)
"""Oblique Procrustes rotation"""


def procrustes_obl(X, *, ndim=2):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Oblique Procrustes rotation

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.msprb.procrustes_obl is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


proc = procrustes_obl


def cheatsheet() -> str:
    return "procrustes_obl({}) -> Oblique Procrustes rotation"


# compact alias per ledger/NAMING.md
procrustesobl = procrustes_obl
