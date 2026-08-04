# morie.fn -- function file (rootcoder007/morie)
"""Online key quantizer: TurboQuant_mse on a single vector."""

import math

from . import _tail1core as C
from . import _b1turbo as B

from ._richresult import RichResult

__all__ = ["kvquant", "turboquant_online_key_quantizer"]


def kvquant(x, b=2, seed=1):
    """Quantize one vector: rotate, scalar-quantize, dequantize, rotate back.

    ONLINE is the property that matters: the rotation and the codebook
    are fixed in advance, so a vector is quantized the moment it
    arrives with no knowledge of the ones that follow.  That is what
    makes this usable on a growing cache, and it is why the codebook
    is built for the ASYMPTOTIC coordinate distribution rather than
    fitted to the data.

    The codebook is Lloyd-Max for a standard normal, rescaled by
    ||x||/sqrt(d): after a rotation the coordinates of a vector of
    length ||x|| are close to i.i.d. N(0, ||x||^2/d).

    Formula: y <- Pi . x;  idx_j <- argmin_k |y_j - c_k|;
             ytilde_j <- c_{idx_j};  xtilde <- Pi' . ytilde

    Parameters
    ----------
    x : array-like
        The vector to quantize.
    b : int
        Bits per coordinate, b >= 1.
    seed : int
        Seed for the pinned rotation.

    Returns
    -------
    RichResult
        ``idx`` (zero-based codebook indices), ``reconstruction``,
        ``codebook``, ``mse``, ``relative_mse``, ``bound`` (Theorem 1),
        ``within_bound``, ``d``, ``b``.

    References
    ----------
    Zandieh et al., TurboQuant: Online Vector Quantization with
    Near-optimal Distortion Rate, arXiv:2504.19874, Algorithm 1
    (TurboQuant_mse) lines 2-11 verbatim, with the Theorem 1 bound
    D_mse <= (sqrt(3) pi / 2) 4^-b.  Fetched from arXiv.  The paper
    specifies the codebook only as the MSE-minimising centroids; the
    Lloyd-Max construction used here is documented in the batch helper
    ``_b1turbo.codebook``.
    """
    x = C.vec(x)
    d = len(x)
    b = int(b)
    if d < 1:
        raise ValueError("the vector must be non-empty")
    if b < 1:
        raise ValueError("the bit width must be at least 1")
    Pi = B.rotation(d, seed)
    y = C.matvec(Pi, x)
    nrm = math.sqrt(sum(v * v for v in x))
    sc = nrm / math.sqrt(d) if nrm > 0 else 1.0
    cb = [sc * v for v in B.codebook(b)]
    idx = B.quantize(y, cb)
    yt = [cb[k] for k in idx]
    xt = C.matvec(C.transpose(Pi), yt)
    mse = sum((x[j] - xt[j]) ** 2 for j in range(d))
    bnd = math.sqrt(3.0) * math.pi / 2.0 * 4.0 ** (-b)
    rel = mse / (nrm * nrm) if nrm > 0 else float("nan")
    return RichResult(payload={
        "idx": [float(v) for v in idx], "reconstruction": xt,
        "codebook": cb, "mse": mse, "relative_mse": rel, "bound": bnd,
        "within_bound": 1.0 if rel <= bnd else 0.0, "d": float(d),
        "b": float(b),
        "method": "TurboQuant_mse, arXiv:2504.19874 Algorithm 1"})


turboquant_online_key_quantizer = kvquant


def cheatsheet():
    return "tqalg1: y = Pi x; idx = nearest centroid; xtilde = Pi^T c[idx]"
