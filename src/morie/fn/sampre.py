"""SAM prompt encoder (points/boxes/masks)."""

__all__ = ["sam_prompt_encoder"]


def sam_prompt_encoder(prompts):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    SAM prompt encoder (points/boxes/masks)

    Formula: learned embeddings per prompt type

    Parameters
    ----------
    prompts : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kirillov et al (2023)
    """
    raise NotImplementedError(
        "morie.fn.sampre.sam_prompt_encoder is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "sampre: SAM prompt encoder (points/boxes/masks)"
