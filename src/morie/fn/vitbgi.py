"""ViT-B/16 initialization (768 dim, 12 heads, 12 layers)."""

__all__ = ["vit_b16_init"]


def vit_b16_init(model):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    ViT-B/16 initialization (768 dim, 12 heads, 12 layers)

    Formula: truncated normal(std=0.02)

    Parameters
    ----------
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dosovitskiy et al (2020)
    """
    raise NotImplementedError(
        "morie.fn.vitbgi.vit_b16_init is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "vitbgi: ViT-B/16 initialization (768 dim, 12 heads, 12 layers)"


# compact alias per ledger/NAMING.md
vitb16init = vit_b16_init
