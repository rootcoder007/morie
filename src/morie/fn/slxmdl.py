"""Spatial Lag of X (SLX) model."""

__all__ = ["slx_model"]


def slx_model(y, X, W):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Spatial Lag of X (SLX) model

    Formula: y = X beta + W X theta + eps

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    W : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Halleck Vega-Elhorst (2015)
    """
    raise NotImplementedError(
        "morie.fn.slxmdl.slx_model is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "slxmdl: Spatial Lag of X (SLX) model"


# compact alias per ledger/NAMING.md
slxmodel = slx_model
