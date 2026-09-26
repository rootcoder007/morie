"""Aitchison compositional spatial"""


def aitchison_sp(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Aitchison compositional spatial

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.zxait.aitchison_sp is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


aitc = aitchison_sp


def cheatsheet() -> str:
    return "aitchison_sp({}) -> Aitchison compositional spatial"


# compact alias per ledger/NAMING.md
aitchisonsp = aitchison_sp
