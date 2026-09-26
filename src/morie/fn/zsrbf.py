"""Multiquadric RBF interpolation"""


def rbf_multiquad(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Multiquadric RBF interpolation

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zsrbf.rbf_multiquad is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


rbf_ = rbf_multiquad


def cheatsheet() -> str:
    return "rbf_multiquad({}) -> Multiquadric RBF interpolation"


# compact alias per ledger/NAMING.md
rbfmultiquad = rbf_multiquad
