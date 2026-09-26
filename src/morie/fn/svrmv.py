"""Rabinowitz-Macdonald intensity component"""


def rm_intensity(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Rabinowitz-Macdonald intensity component

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svrmv.rm_intensity is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


rm_i = rm_intensity


def cheatsheet() -> str:
    return "rm_intensity({}) -> Rabinowitz-Macdonald intensity component"


# compact alias per ledger/NAMING.md
rmintensity = rm_intensity
