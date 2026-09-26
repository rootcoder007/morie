"""Three-way spatial tensor"""


def tensor_3way_sp(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Three-way spatial tensor

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.zxtn3.tensor_3way_sp is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


tens = tensor_3way_sp


def cheatsheet() -> str:
    return "tensor_3way_sp({}) -> Three-way spatial tensor"


# compact alias per ledger/NAMING.md
tensor3waysp = tensor_3way_sp
