"""SAM image encoder (ViT-H)."""

__all__ = ["sam_image_encoder"]


def sam_image_encoder(image):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    SAM image encoder (ViT-H)

    Formula: large ViT pretrained on SA-1B

    Parameters
    ----------
    image : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kirillov et al (2023) SAM
    """
    raise NotImplementedError(
        "morie.fn.semaeg.sam_image_encoder is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "semaeg: SAM image encoder (ViT-H)"
