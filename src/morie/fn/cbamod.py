"""CBAM channel + spatial attention."""

__all__ = ["cbam_attention"]


def cbam_attention(x):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    CBAM channel + spatial attention

    Formula: channel attn × spatial attn

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Woo et al (2018) CBAM
    """
    raise NotImplementedError(
        "morie.fn.cbamod.cbam_attention is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "cbamod: CBAM channel + spatial attention"


# compact alias per ledger/NAMING.md
cbamattention = cbam_attention
