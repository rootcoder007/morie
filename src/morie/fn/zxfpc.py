"""Spatial functional PCA"""


def fpca_spatial(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Spatial functional PCA

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.zxfpc.fpca_spatial is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


fpca = fpca_spatial


def cheatsheet() -> str:
    return "fpca_spatial({}) -> Spatial functional PCA"


# compact alias per ledger/NAMING.md
fpcaspatial = fpca_spatial
