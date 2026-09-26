"""CLIP text encoder (transformer)."""

__all__ = ["clip_text_encoder"]


def clip_text_encoder(text):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    CLIP text encoder (transformer)

    Formula: BPE tokenize + transformer + [EOT] embed

    Parameters
    ----------
    text : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Radford et al (2021)
    """
    raise NotImplementedError(
        "morie.fn.clipxt.clip_text_encoder is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "clipxt: CLIP text encoder (transformer)"
