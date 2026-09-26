"""Shapley value in spatial game"""


def shapley_spatial(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Shapley value in spatial game

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svshp.shapley_spatial is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


shap = shapley_spatial


def cheatsheet() -> str:
    return "shapley_spatial({}) -> Shapley value in spatial game"


# compact alias per ledger/NAMING.md
shapleyspatial = shapley_spatial
