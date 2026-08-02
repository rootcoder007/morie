# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sinusoidal positional encoding (Vaswani et al. 2017)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_sinusoidal_positional_encoding"]

_METHOD = "Sinusoidal positional encoding"


def geron_sinusoidal_positional_encoding(seq_len, d_model, base=10000.0):
    r"""Fixed position signal added to token embeddings.

    .. math::
        PE(\text{pos}, 2i) = \sin\!\left(\frac{\text{pos}}{10000^{2i/d}}\right),
        \qquad
        PE(\text{pos}, 2i+1) = \cos\!\left(\frac{\text{pos}}{10000^{2i/d}}\right)

    Attention is permutation-equivariant, so without this the model
    cannot tell "dog bites man" from "man bites dog".  The geometric
    spread of wavelengths -- :math:`2\pi` up to :math:`2\pi \cdot 10^4` --
    means each dimension resolves a different distance scale, and because
    :math:`PE_{\text{pos}+k}` is a fixed linear map of
    :math:`PE_{\text{pos}}` (a rotation per frequency pair), relative
    offsets are linearly recoverable.  Nothing is learned, so the encoding
    extrapolates past the training length.

    Parameters
    ----------
    seq_len : int
        Number of positions, at least 1.
    d_model : int
        Embedding width; must be even so every sine has its cosine.
    base : float, optional
        Wavelength base, default 10000.

    Returns
    -------
    RichResult
        Payload keys ``encoding`` (seq_len x d_model), ``wavelengths``,
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 15, Positional Encodings section.

    Examples
    --------
    Position 0 is ``sin 0 = 0`` and ``cos 0 = 1`` alternating:

    >>> r = geron_sinusoidal_positional_encoding(3, 4)
    >>> [round(v, 6) for v in r["encoding"][0]]
    [0.0, 1.0, 0.0, 1.0]

    Position 1, dimension 0 is ``sin(1) = 0.841471``; dimension 2 runs at
    wavelength ``10000^0.5 = 100``, so it is ``sin(0.01)``:

    >>> [round(v, 6) for v in r["encoding"][1]]
    [0.841471, 0.540302, 0.01, 0.99995]

    Every row has the same norm ``sqrt(d/2)``:

    >>> import numpy as np
    >>> round(float(np.linalg.norm(r["encoding"][2])), 10)
    1.4142135624
    """
    seq_len = int(seq_len)
    d_model = int(d_model)
    if seq_len < 1:
        raise ValueError(f"seq_len must be at least 1, got {seq_len}.")
    if d_model < 2:
        raise ValueError(f"d_model must be at least 2, got {d_model}.")
    if d_model % 2 != 0:
        raise ValueError(
            f"d_model must be even so each sine has a paired cosine, got {d_model}."
        )
    base = float(base)
    if not np.isfinite(base) or base <= 1:
        raise ValueError(f"base must be a finite float greater than 1, got {base}.")

    pos = np.arange(seq_len, dtype=float)[:, None]
    i = np.arange(d_model // 2, dtype=float)[None, :]
    div = base ** (2.0 * i / d_model)
    ang = pos / div
    PE = np.empty((seq_len, d_model))
    PE[:, 0::2] = np.sin(ang)
    PE[:, 1::2] = np.cos(ang)

    return RichResult(
        title="Sinusoidal positional encoding",
        summary_lines=[("Positions", seq_len), ("d_model", d_model)],
        payload={
            "encoding": PE.tolist(),
            "wavelengths": (2 * np.pi * div.ravel()).tolist(),
            "estimate": PE.tolist(),
            "n": seq_len,
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grpe: PE(pos,2i)=sin(pos/10000^(2i/d)), PE(pos,2i+1)=cos(...); fixed, extrapolates, row norm sqrt(d/2)"
