"""CLIP image-text contrastive alignment."""

__all__ = ["clip_image_text_align"]


def clip_image_text_align(images, texts, tau):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    CLIP image-text contrastive alignment

    Formula: -1/N sum log exp(I_i^T T_i / tau)/sum_j ...

    Parameters
    ----------
    images : array-like
        Input data.
    texts : array-like
        Input data.
    tau : array-like
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
        "morie.fn.clipbn.clip_image_text_align is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "clipbn: CLIP image-text contrastive alignment"
