"""2D valence spatial model"""


def valence_2d(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    2D valence spatial model

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svvl2.valence_2d is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


vale = valence_2d


def cheatsheet() -> str:
    return "valence_2d({}) -> 2D valence spatial model"


# compact alias per ledger/NAMING.md
valence2d = valence_2d
