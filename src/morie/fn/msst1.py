# morie.fn -- function file (rootcoder007/morie)
"""Raw stress (Kruskal stress-1)"""


def stress_raw(X, *, ndim=2):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Raw stress (Kruskal stress-1)

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.msst1.stress_raw is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


stre = stress_raw


def cheatsheet() -> str:
    return "stress_raw({}) -> Raw stress (Kruskal stress-1)"


# compact alias per ledger/NAMING.md
stressraw = stress_raw
