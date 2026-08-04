# morie.fn -- function file (rootcoder007/morie)
"""Inner-product (score) distortion of a quantized cache."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["scoredist", "turboquant_score_distortion"]


def scoredist(b, d, query_norm=1.0, n_keys=1):
    """Distortion of an inner-product score under TurboQuant_prod.

    The 1/d in the bound is what makes this useful: the inner-product
    distortion falls with DIMENSION as well as with bits, so a long
    key vector is cheap to quantize even though the vector-space error
    is not small.  Reconstructing the key and then taking the inner
    product would not have that property.

    ``expected_max`` is a Gaussian-tail estimate of the largest of
    ``n_keys`` independent score errors -- it is an ESTIMATE, not a
    bound from the paper, and is labelled as such.

    Formula: D_prod <= sqrt(3) pi^2 ||q||^2 / d . 4^-b  (Theorem 2);
             rms error = sqrt(D_prod);
             expected max of n errors ~= rms sqrt(2 log n)

    Parameters
    ----------
    b : float
        Bits per coordinate.
    d : int
        Key dimension.
    query_norm : float
        ||q||_2 of the query.
    n_keys : int
        Number of keys the query is scored against.

    Returns
    -------
    RichResult
        ``variance`` (D_prod), ``rms``, ``lower_bound``, ``ratio``,
        ``expected_max``, ``b``, ``d``, ``n_keys``.

    References
    ----------
    Zandieh et al., TurboQuant: Online Vector Quantization with
    Near-optimal Distortion Rate, arXiv:2504.19874, Theorem 2:
    E[<y, xtilde>] = <y, x> and D_prod <= (sqrt(3) pi^2 ||y||_2^2 / d)
    . 1/4^b; and Theorem 3 for the matching lower bound
    D_prod(Q) >= (1/d) . 1/4^b.  Fetched from arXiv.  The
    ``expected_max`` figure is the standard Gaussian maximum
    approximation and is NOT from the paper.
    """
    b = float(b)
    d = int(d)
    qn = float(query_norm)
    nk = int(n_keys)
    if b < 0:
        raise ValueError("the bit-width must be non-negative")
    if d < 1:
        raise ValueError("the dimension must be at least 1")
    if qn < 0:
        raise ValueError("the query norm must be non-negative")
    if nk < 1:
        raise ValueError("there must be at least one key")
    q = 4.0 ** (-b)
    var = math.sqrt(3.0) * math.pi * math.pi * qn * qn / d * q
    lo = q / d
    rms = math.sqrt(var)
    em = rms * math.sqrt(2.0 * math.log(nk)) if nk > 1 else rms
    return RichResult(payload={
        "variance": var, "rms": rms, "lower_bound": lo,
        "ratio": var / lo if lo > 0 else float("nan"),
        "expected_max": em, "b": b, "d": float(d), "n_keys": float(nk),
        "method": "Score distortion, arXiv:2504.19874 Theorems 2 and 3"})


turboquant_score_distortion = scoredist


def cheatsheet():
    return "tqscr: D_prod <= sqrt(3)pi^2 ||q||^2/(d 4^b); falls with d as well as b"
