# morie.fn -- function file (rootcoder007/morie)
"""Normalized stress"""


def stress_norm(X, *, ndim=2):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Normalized stress

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.msst2.stress_norm is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


stre = stress_norm


def cheatsheet() -> str:
    return "stress_norm({}) -> Normalized stress"


# compact alias per ledger/NAMING.md
stressnorm = stress_norm
