# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Root-mean-square layer normalisation (RMSNorm)."""

from . import _array_core as np

from ._richresult import RichResult
from .rmsnr import rms_norm

__all__ = ["kamath_rms_norm"]


def kamath_rms_norm(x, g=None, eps=1e-6):
    """RMSNorm(x) = x / RMS(x) * g, RMS(x) = sqrt(mean(x^2) + eps).

    Kamath's RMSNorm is Zhang and Sennrich's RMSNorm, so the
    computation is DELEGATED to ``morie.fn.rmsnr`` rather than kept as
    a second copy. Unlike LayerNorm there is no mean subtraction and
    no bias: only the scale is removed, which is the whole saving.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, RMSNorm (Zhang and
    Sennrich 2019).

    Examples
    --------
    >>> out = kamath_rms_norm([3.0, 4.0], eps=0.0)
    >>> abs(out["y"][0] - 3.0 / 12.5 ** 0.5) < 1e-12
    True
    >>> abs(out["rms"][0] - 12.5 ** 0.5) < 1e-12
    True
    >>> scaled = kamath_rms_norm([3.0, 4.0], g=[2.0, 2.0], eps=0.0)
    >>> abs(scaled["y"][1] - 2 * 4.0 / 12.5 ** 0.5) < 1e-12
    True
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("nothing to normalise.")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must be finite.")
    eps = float(eps)
    if eps < 0:
        raise ValueError("eps must be non-negative.")
    if g is not None:
        gg = np.atleast_1d(np.asarray(g, dtype=float)).ravel()
        if gg.size != x.shape[-1]:
            raise ValueError(
                f"the gain has {gg.size} entries for a feature axis of "
                f"{x.shape[-1]}.")
    if eps == 0.0 and np.all(np.atleast_2d(x) == 0):
        raise ValueError(
            "RMS(x) is 0 with eps = 0, so the normalisation divides by "
            "zero; pass a positive eps.")
    base = rms_norm(x, gamma=g, eps=eps)
    y = np.asarray(base["tensor"], dtype=float)
    rms = np.atleast_1d(np.asarray(base["rms"], dtype=float)).ravel()
    return RichResult(payload={
        "y": [float(v) for v in np.atleast_1d(y).ravel()],
        "tensor": y,
        "rms": [float(v) for v in rms],
        "estimate": float(np.atleast_1d(y).ravel()[0]),
        "eps": eps, "n": int(x.shape[-1]),
        "method": "RMSNorm x / sqrt(mean(x^2) + eps) * g "
                  "(delegates to rmsnr)"})


def cheatsheet():
    return "kmrmsn: x/sqrt(mean(x^2)+eps)*g via rmsnr; no mean, no bias"


# compact alias per ledger/NAMING.md
kamathrmsnorm = kamath_rms_norm
