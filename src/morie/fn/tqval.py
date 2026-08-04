# morie.fn -- function file (rootcoder007/morie)
"""Value-cache quantization by the inner-product-optimized scheme."""

import math

from . import _tail1core as C
from . import _b1turbo as B

from ._richresult import RichResult

__all__ = ["vcquant", "turboquant_value_cache_quantization"]


def vcquant(V, b=3, seed=1):
    """Quantize a value cache with the inner-product variant of TurboQuant.

    Values are consumed by inner products, not reconstructed for their
    own sake, so the right objective is the inner-product distortion
    and not the MSE -- which is exactly why the paper carries two
    algorithms.  One bit per coordinate is spent on the sign sketch of
    the RESIDUAL, so the scalar stage only gets b - 1: that is the
    trade, and it is why b must be at least 2.

    Formula: idx <- Quant_mse(x) at b - 1 bits;  r <- x - DeQuant_mse(idx);
             qjl <- sign(S . r);
             xtilde <- DeQuant_mse(idx) + (sqrt(pi/2)/d) ||r|| S' qjl

    Parameters
    ----------
    V : array-like, shape (n, d)
        Value cache, one value vector per row.
    b : int
        Total bits per coordinate, b >= 2.
    seed : int
        Seed for the pinned rotation and projection.

    Returns
    -------
    RichResult
        ``reconstruction`` (n x d), ``mse`` (per row),
        ``residual_norm`` (per row), ``mean_mse``, ``n``, ``d``,
        ``b``.

    References
    ----------
    Zandieh et al., TurboQuant: Online Vector Quantization with
    Near-optimal Distortion Rate, arXiv:2504.19874, Algorithm 2
    (TurboQuant_prod) lines 2-12: instantiate TurboQuant_mse with
    bit-width b - 1, take the residual r <- x - DeQuant_mse(idx),
    sketch it with qjl <- sign(S . r), and reconstruct as
    xtilde_mse + (sqrt(pi/2)/d) gamma S' qjl.  Fetched from arXiv.
    """
    V = C.mat(V)
    n = len(V)
    if n < 1:
        raise ValueError("the cache must hold at least one value vector")
    d = len(V[0])
    if any(len(r) != d for r in V):
        raise ValueError("every value vector must have the same dimension")
    b = int(b)
    if b < 2:
        raise ValueError(
            "b must be at least 2: one bit per coordinate goes to the "
            "sign sketch of the residual")
    Pi = B.rotation(d, seed)
    Pit = C.transpose(Pi)
    base = B.codebook(b - 1)
    g = C.Lcg(seed + 1)
    S = [[g.norm() for _ in range(d)] for _ in range(d)]
    rec = []
    mse = []
    rn = []
    for i in range(n):
        x = V[i]
        y = C.matvec(Pi, x)
        nrm = math.sqrt(sum(v * v for v in x))
        sc = nrm / math.sqrt(d) if nrm > 0 else 1.0
        cb = [sc * v for v in base]
        yt = [cb[k] for k in B.quantize(y, cb)]
        xm = C.matvec(Pit, yt)
        r = [x[j] - xm[j] for j in range(d)]
        gam = math.sqrt(sum(v * v for v in r))
        q = [1.0 if v >= 0.0 else -1.0 for v in C.matvec(S, r)]
        k = math.sqrt(math.pi / 2.0) / d * gam
        xt = [xm[j] + k * sum(S[t][j] * q[t] for t in range(d))
              for j in range(d)]
        rec.append(xt)
        mse.append(sum((x[j] - xt[j]) ** 2 for j in range(d)))
        rn.append(gam)
    return RichResult(payload={
        "reconstruction": rec, "mse": mse, "residual_norm": rn,
        "mean_mse": sum(mse) / n, "n": float(n), "d": float(d),
        "b": float(b),
        "method": "TurboQuant_prod on a value cache, arXiv:2504.19874"})


turboquant_value_cache_quantization = vcquant


def cheatsheet():
    return "tqval: Alg 2 - scalar stage at b-1 bits + sign sketch of the residual"
