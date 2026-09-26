"""Temporal Fusion Transformer."""

__all__ = ["temporal_fusion_transformer"]


def temporal_fusion_transformer(X, y, static_cov):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Temporal Fusion Transformer

    Formula: variable-selection + LSTM encoder + multi-head attn

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    static_cov : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lim et al (2021) TFT
    """
    raise NotImplementedError(
        "morie.fn.tft.temporal_fusion_transformer is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "tft: Temporal Fusion Transformer"
