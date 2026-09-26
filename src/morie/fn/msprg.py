# morie.fn -- function file (rootcoder007/morie)
"""Generalized Procrustes analysis"""


def procrustes_gen(X, *, ndim=2):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Generalized Procrustes analysis

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.msprg.procrustes_gen is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


proc = procrustes_gen


def cheatsheet() -> str:
    return "procrustes_gen({}) -> Generalized Procrustes analysis"


# compact alias per ledger/NAMING.md
procrustesgen = procrustes_gen
