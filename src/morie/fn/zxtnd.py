"""Spatial tensor decomposition"""


def tensor_decomp_sp(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Spatial tensor decomposition

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.zxtnd.tensor_decomp_sp is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


tens = tensor_decomp_sp


def cheatsheet() -> str:
    return "tensor_decomp_sp({}) -> Spatial tensor decomposition"


# compact alias per ledger/NAMING.md
tensordecompsp = tensor_decomp_sp
