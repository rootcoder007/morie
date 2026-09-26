# morie.fn -- function file (rootcoder007/morie)
"""GWR corrected AIC (AICc) for model selection."""


def gwraicc(ll, k, n):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    GWR corrected AIC (AICc) for model selection.

    Category: GWR

    Parameters
    ----------
    ll, k, n : see function signature.

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.gwraicc.gwraicc is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


gwraicc_fn = gwraicc


def cheatsheet() -> str:
    return "gwraicc({}) -> GWR corrected AIC (AICc) for model selection."
