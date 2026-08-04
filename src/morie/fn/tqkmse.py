# morie.fn -- function file (rootcoder007/morie)
"""Mean squared error of a quantized key cache."""

import math

from . import _tail1core as C
from . import _b1turbo as B

from ._richresult import RichResult

__all__ = ["kvmse", "turboquant_kv_mse"]


def kvmse(K, b=2, seed=1):
    """Quantize every key in a cache and measure the achieved MSE.

    The per-key MSE is reported alongside the mean, because a cache is
    only as good as its worst key: one badly reconstructed key can
    dominate an attention score even when the average distortion looks
    fine.  ``worst_relative`` is that key.

    The rotation is shared across keys -- which is the point of the
    online scheme, since it must be fixed before any key arrives -- so
    the same Pi is applied to every row.

    Formula: for each row k_i, xtilde_i = TurboQuant_mse(k_i, b);
             MSE_i = ||k_i - xtilde_i||^2, relative to ||k_i||^2

    Parameters
    ----------
    K : array-like, shape (n, d)
        Key cache, one key per row.
    b : int
        Bits per coordinate.
    seed : int
        Seed for the shared pinned rotation.

    Returns
    -------
    RichResult
        ``mse`` (per key), ``relative_mse`` (per key), ``mean_mse``,
        ``mean_relative``, ``worst_relative``, ``bound``,
        ``within_bound``, ``n``, ``d``, ``b``.

    References
    ----------
    Zandieh et al., TurboQuant: Online Vector Quantization with
    Near-optimal Distortion Rate, arXiv:2504.19874, Algorithm 1 and
    Theorem 1 (D_mse <= (sqrt(3) pi / 2) 4^-b).  Fetched from arXiv.
    NOTE: the worklist filed this row under
    "vdLaan-ICLR2026-arxiv-2504.19874"; arXiv 2504.19874 is Zandieh et
    al.'s TurboQuant, not a van der Laan paper, and the attribution has
    been corrected here.
    """
    K = C.mat(K)
    n = len(K)
    if n < 1:
        raise ValueError("the cache must hold at least one key")
    d = len(K[0])
    if any(len(r) != d for r in K):
        raise ValueError("every key must have the same dimension")
    b = int(b)
    if b < 1:
        raise ValueError("the bit width must be at least 1")
    Pi = B.rotation(d, seed)
    Pit = C.transpose(Pi)
    base = B.codebook(b)
    mse = []
    rel = []
    for i in range(n):
        x = K[i]
        y = C.matvec(Pi, x)
        nrm = math.sqrt(sum(v * v for v in x))
        sc = nrm / math.sqrt(d) if nrm > 0 else 1.0
        cb = [sc * v for v in base]
        yt = [cb[k] for k in B.quantize(y, cb)]
        xt = C.matvec(Pit, yt)
        e = sum((x[j] - xt[j]) ** 2 for j in range(d))
        mse.append(e)
        rel.append(e / (nrm * nrm) if nrm > 0 else float("nan"))
    bnd = math.sqrt(3.0) * math.pi / 2.0 * 4.0 ** (-b)
    mr = sum(rel) / n
    return RichResult(payload={
        "mse": mse, "relative_mse": rel, "mean_mse": sum(mse) / n,
        "mean_relative": mr, "worst_relative": max(rel), "bound": bnd,
        "within_bound": 1.0 if mr <= bnd else 0.0, "n": float(n),
        "d": float(d), "b": float(b),
        "method": "Key-cache MSE under TurboQuant_mse, arXiv:2504.19874"})


turboquant_kv_mse = kvmse


def cheatsheet():
    return "tqkmse: per-key MSE of TurboQuant_mse against the Theorem 1 bound"
