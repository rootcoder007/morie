"""Banzhaf power index spatial"""


def banzhaf_spatial(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Banzhaf power index spatial

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svbnz.banzhaf_spatial is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


banz = banzhaf_spatial


def cheatsheet() -> str:
    return "banzhaf_spatial({}) -> Banzhaf power index spatial"


# compact alias per ledger/NAMING.md
banzhafspatial = banzhaf_spatial
