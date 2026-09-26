"""Spatial Gaussian mixture"""


def gmm_spatial(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Spatial Gaussian mixture

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.zxgmm.gmm_spatial is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


gmm_ = gmm_spatial


def cheatsheet() -> str:
    return "gmm_spatial({}) -> Spatial Gaussian mixture"


# compact alias per ledger/NAMING.md
gmmspatial = gmm_spatial
