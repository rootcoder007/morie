# morie.fn -- function file (rootcoder007/morie)
"""VQ-VAE codebook lookup with commitment loss."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_vqvae_quantize", "geron_vq_vae_codebook_loss"]


def geron_vqvae_quantize(z_e, codebook, beta=0.25):
    r"""Nearest-codebook quantisation with the two VQ-VAE losses.

    .. math::
       z_q = e_k,\quad k = \arg\min_j \lVert z_e - e_j \rVert_2,

    .. math::
       L = \lVert \mathrm{sg}[z_e] - e \rVert^2
           + \beta \lVert z_e - \mathrm{sg}[e] \rVert^2 ,

    with :math:`\mathrm{sg}` the stop-gradient. The two terms look
    symmetric and are not: the FIRST moves the codebook toward the
    encoder, the SECOND (the commitment loss) moves the encoder toward
    the codebook, and :math:`\beta` sets how much the encoder is
    allowed to wander. Without the commitment term the encoder output
    can grow without bound, because nothing penalises it for
    outrunning a codebook that is chasing it.

    The argmin has no gradient, so training uses the STRAIGHT-THROUGH
    estimator: the forward pass emits :math:`z_q`, the backward pass
    copies the gradient from :math:`z_q` straight to :math:`z_e` as
    though quantisation were the identity. ``straight_through`` returns
    the value that carries this, :math:`z_e + \mathrm{sg}[z_q - z_e]`,
    which equals :math:`z_q` numerically and differentiates as
    :math:`z_e`.

    ``perplexity`` measures codebook usage,
    :math:`\exp(-\sum_j p_j\log p_j)`. Codebook COLLAPSE -- most
    entries never selected -- is the standard failure, and a
    perplexity far below the codebook size is what it looks like.

    Parameters
    ----------
    z_e : array-like, shape (n, d)
        Encoder outputs.
    codebook : array-like, shape (K, d)
    beta : float
        Commitment weight; 0.25 in the original paper.

    Returns
    -------
    RichResult
        ``z_q``, ``indices``, ``codebook_loss``, ``commitment_loss``,
        ``loss``, ``straight_through``, ``perplexity``,
        ``used_codes``, ``collapse``.

    References
    ----------
    Geron (2022), *Hands-On Machine Learning*, 3rd ed., chapter 17.
    van den Oord, Vinyals and Kavukcuoglu (2017), "Neural discrete
    representation learning", NeurIPS.

    Examples
    --------
    >>> out = geron_vqvae_quantize([[0.1, 0.0]], [[0.0, 0.0], [1.0, 1.0]])
    >>> int(out["indices"][0])
    0
    """
    Z = np.atleast_2d(np.asarray(z_e, dtype=float))
    E = np.atleast_2d(np.asarray(codebook, dtype=float))
    if Z.shape[1] != E.shape[1]:
        raise ValueError(
            "z_e has dimension %d, codebook has %d."
            % (Z.shape[1], E.shape[1])
        )
    n, K = Z.shape[0], E.shape[0]
    if K < 1:
        raise ValueError("codebook is empty.")
    if beta < 0:
        raise ValueError("beta must be non-negative, got %r." % beta)

    d2 = ((Z[:, None, :] - E[None, :, :]) ** 2).sum(axis=2)
    idx = np.argmin(d2, axis=1)
    zq = E[idx]

    cb = float(np.mean(np.sum((zq - Z) ** 2, axis=1)))       # updates codebook
    commit = float(np.mean(np.sum((Z - zq) ** 2, axis=1)))   # updates encoder
    counts = np.bincount(idx, minlength=K).astype(float)
    p = counts / counts.sum()
    nz = p[p > 0]
    perp = float(np.exp(-np.sum(nz * np.log(nz))))
    return RichResult(
        payload={
            "estimate": zq,
            "z_q": zq,
            "indices": idx,
            "codebook_loss": cb,
            "commitment_loss": commit,
            "loss": float(cb + beta * commit),
            "beta": float(beta),
            "loss_note": (
                "the codebook term pulls the codebook toward the encoder, "
                "the commitment term pulls the encoder toward the codebook; "
                "drop the second and the encoder output can grow without "
                "bound"
            ),
            "straight_through": zq,
            "straight_through_note": (
                "forward value is z_q, backward gradient is copied to z_e as "
                "though quantisation were the identity -- argmin has no "
                "gradient of its own"
            ),
            "perplexity": perp,
            "used_codes": int(np.sum(counts > 0)),
            "codebook_size": int(K),
            "collapse": bool(perp < 0.5 * K),
            "collapse_note": (
                "perplexity far below the codebook size means most entries "
                "are never selected -- the standard VQ-VAE failure"
            ),
            "n": int(n),
            "method": "VQ-VAE nearest-codebook quantisation",
        }
    )


def cheatsheet():
    return (
        "grvqv: VQ-VAE quantisation with codebook and commitment losses, "
        "straight-through gradient and a collapse check"
    )


#: Catalogue alias for :func:`geron_vqvae_quantize`.
geron_vq_vae_codebook_loss = geron_vqvae_quantize
