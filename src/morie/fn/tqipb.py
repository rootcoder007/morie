"""Inner-product distortion bounds for TurboQuant-packed blocks.

TurboQuant (Zandieh, Han, Daliri and Karbasi, 2025) quantizes a vector by
randomly rotating it first. The rotation is what makes the scheme
analysable: a random rotation of any vector on the unit sphere has
coordinates distributed Beta((d-1)/2, (d-1)/2) rescaled to [-1, 1], the
same law for every input, so ONE scalar quantizer designed against that
one distribution is optimal for every input rather than for a training
set. The distortion constants below are that quantizer's, not a fit.

What the paper gives, and what this module does with it, are separate
things and are kept separate here.

From the paper
  D_prod <= sqrt(3) * pi^2 * ||y||^2 / (d * 4^b)          (Theorem 2)
      the inner-product distortion of TurboQuant_prod at bit-width b in
      dimension d, for x on the unit sphere. It is the high-resolution
      (Panter-Dite) form, so it is loose at b = 1 and tightens as b
      grows.
  D_prod >= ||y||^2 / (d * 4^b)                           (Theorem 3)
      no randomized quantizer at bit-width b can do better on the worst
      input. The gap between the two is the constant sqrt(3) * pi^2,
      about 17.1 in this normalisation.
  D_prod ~= {1.57, 0.56, 0.18, 0.047} * ||y||^2 / d       (Table, b=1..4)
      the ACTUAL distortion of the optimal b-bit quantizer for the Beta
      law, obtained by solving the continuous 1-D k-means problem rather
      than by the high-resolution approximation. At b = 1 this is 1.57
      against Panter-Dite's 4.27, which is the size of the error the
      asymptotic form makes at one bit.
  Var <= pi * ||y||^2 / (2 * d)                           (Lemma 4)
      the one-bit Quantized Johnson-Lindenstrauss transform, which
      TurboQuant_prod applies to the residual to make the estimate
      unbiased.

Not from the paper
  The paper states distortions -- second moments -- and does NOT state a
  concentration inequality of the form

      Pr[ |<x_hat, y_hat> - <x, y>| > eps ||x|| ||y|| ] <= delta.

  Turning a variance into such a statement needs a tail inequality, and
  which one is a modelling choice, so both are offered and the choice is
  reported in the result:

    "chebyshev"     delta = V / (eps^2 ||x||^2 ||y||^2). Assumption-free
                    given the variance, and correspondingly weak.
    "sub_gaussian"  delta = 2 exp(-eps^2 ||x||^2 ||y||^2 / (2 V)).
                    Requires the error to be sub-Gaussian with variance
                    proxy V. TurboQuant's error is a sum of d bounded
                    terms after the rotation, which is the usual reason
                    to reach for this, but it is an assumption and is
                    labelled as one.

  Neither is attributed to the paper. `route` selects the variance;
  `tail` selects the inequality; both travel in the payload.

Packed blocks
  A long vector is quantized in blocks, each rotated and quantized on
  its own, and the inner product is the sum of the per-block inner
  products. The per-block errors are independent -- separate rotations
  -- so the VARIANCES add:

      V = sum_j  c(b, d_j) * ||x_j||^2 * ||y_j||^2

  with c(b, d_j) the per-coordinate constant of the chosen route. Under
  the proportional split below this is EXACTLY neutral: every route's
  constant goes as 1/d_j while a block of d_j coordinates carries
  (d_j/d)^2 of the norm product, so the two cancel and the total is
  c(b, d) ||x||^2 ||y||^2 whatever the blocking. That is a statement
  about the bound, not about an implementation -- the distortion is per
  coordinate, so how the coordinates are grouped does not enter it. The
  sum is computed rather than the cancellation asserted, and the parity
  harness anchors on it.

Reference
  Zandieh, A., Han, I., Daliri, M. and Karbasi, A. (2025) "TurboQuant:
    Online Vector Quantization with Near-optimal Distortion Rate."
    arXiv:2504.19874. Theorem 2 (upper bound), Theorem 3 (lower bound),
    Lemma 4 (QJL variance), and the b = 1..4 distortion table.
"""

import math

from ._richresult import RichResult

__all__ = ["turboquant_inner_product_distortion_bound", "tqipb",
           "distortion_constant", "variance_bound", "tail_probability",
           "bits_required", "cheatsheet", "ROUTES", "TAILS"]

ROUTES = ("table", "panter_dite", "qjl", "lower_bound")
TAILS = ("chebyshev", "sub_gaussian")

# Solved 1-D k-means distortions for the Beta law, b = 1..4 (paper table).
_TABLE = {1: 1.57, 2: 0.56, 3: 0.18, 4: 0.047}

# sqrt(3) * pi^2, the Panter-Dite constant in this normalisation.
_PD = math.sqrt(3.0) * math.pi * math.pi


def _pow4(b):
    """4^b by repeated multiplication.

    Written out because R's `^` on an integer exponent is repeated
    squaring while Python's `**` calls libm pow(); for a quantity this
    small the two agree, but the module does not rely on that.
    """
    p = 1.0
    for _ in range(int(b)):
        p *= 4.0
    return p


def distortion_constant(bits, d, route="table"):
    """Per-unit-norm inner-product distortion constant c(b, d).

    The returned c satisfies E[(<x_hat, y> - <x, y>)^2] <= c ||x||^2
    ||y||^2 for the chosen route.

    Parameters
    ----------
    bits : int
        Bit-width per coordinate, b >= 1.
    d : int
        Block dimension.
    route : str
        One of "table" (the paper's solved b = 1..4 constants, falling
        back to Panter-Dite above 4 bits, which is where the
        high-resolution form is accurate), "panter_dite" (Theorem 2),
        "qjl" (Lemma 4, one bit, independent of b), or "lower_bound"
        (Theorem 3, what no scheme can beat).

    Returns
    -------
    float
    """
    b = int(bits)
    d = int(d)
    if b < 1:
        raise ValueError("bits must be at least 1")
    if d < 1:
        raise ValueError("d must be at least 1")
    if route not in ROUTES:
        raise ValueError("route must be one of %r" % (ROUTES,))
    if route == "qjl":
        return math.pi / (2.0 * d)
    if route == "lower_bound":
        return 1.0 / (d * _pow4(b))
    if route == "table" and b in _TABLE:
        return _TABLE[b] / d
    return _PD / (d * _pow4(b))


def _blocks_of(d, n_blocks):
    """Split d coordinates into n_blocks as evenly as possible.

    The first `d % n_blocks` blocks get one extra coordinate, which is
    what a packer does when the dimension does not divide.
    """
    d = int(d)
    k = int(n_blocks)
    if k < 1 or k > d:
        raise ValueError("n_blocks must be in 1..d")
    base = d // k
    rem = d - base * k
    return [base + 1 if j < rem else base for j in range(k)]


def variance_bound(bits, d, norm_sq=1.0, x_norm_sq=1.0, n_blocks=1,
                   route="table"):
    """Variance of the inner-product estimate, summed over packed blocks.

    Mass is split across blocks in proportion to their dimension, which
    is the right split for a randomly rotated vector: after the rotation
    no coordinate is special, so a block of d_j coordinates carries
    d_j / d of the squared norm in expectation.

    Parameters
    ----------
    bits : int
        Bit-width per coordinate.
    d : int
        Total dimension.
    norm_sq : float
        ||y||^2.
    x_norm_sq : float
        ||x||^2. The paper states its bounds for x on the unit sphere;
        the distortion is homogeneous in ||x||^2, so this scales it.
    n_blocks : int
        Number of independently rotated and quantized blocks.
    route : str
        See `distortion_constant`.

    Returns
    -------
    float
    """
    d = int(d)
    dims = _blocks_of(d, n_blocks)
    total = 0.0
    for dj in dims:
        share = float(dj) / float(d)
        total += (distortion_constant(bits, dj, route)
                  * (x_norm_sq * share) * (norm_sq * share))
    return total


def tail_probability(var, eps, norm_sq=1.0, x_norm_sq=1.0,
                     tail="chebyshev"):
    """delta such that Pr[|error| > eps ||x|| ||y||] <= delta.

    Parameters
    ----------
    var : float
        Variance bound, from `variance_bound`.
    eps : float
        Relative accuracy, eps > 0.
    norm_sq, x_norm_sq : float
        ||y||^2 and ||x||^2, so the threshold is in the same units as
        the variance.
    tail : str
        "chebyshev" (no assumption beyond the variance) or
        "sub_gaussian" (assumes the error is sub-Gaussian with variance
        proxy `var`; this is an assumption the paper does not make).

    Returns
    -------
    float
        Clamped to at most 1.0, since a probability bound above 1 says
        nothing.
    """
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if tail not in TAILS:
        raise ValueError("tail must be one of %r" % (TAILS,))
    thresh_sq = eps * eps * x_norm_sq * norm_sq
    if var <= 0.0:
        return 0.0
    if tail == "chebyshev":
        p = var / thresh_sq
    else:
        p = 2.0 * math.exp(-thresh_sq / (2.0 * var))
    return p if p < 1.0 else 1.0


def bits_required(eps, delta, d, norm_sq=1.0, x_norm_sq=1.0, n_blocks=1,
                  route="table", tail="chebyshev", max_bits=32):
    """Smallest bit-width meeting (eps, delta), or None within max_bits.

    Searched rather than inverted in closed form: the "table" route is
    not a closed-form function of b below 5 bits, and a search over at
    most 32 values costs nothing.
    """
    for b in range(1, int(max_bits) + 1):
        v = variance_bound(b, d, norm_sq, x_norm_sq, n_blocks, route)
        if tail_probability(v, eps, norm_sq, x_norm_sq, tail) <= delta:
            return b
    return None


def turboquant_inner_product_distortion_bound(bits, norm_sq=1.0, d=None,
                                              eps=0.1, delta=None,
                                              x_norm_sq=1.0, n_blocks=1,
                                              route="table",
                                              tail="chebyshev"):
    """Inner-product distortion and tail bound for TurboQuant blocks.

    Parameters
    ----------
    bits : int
        Bit-width per coordinate, b >= 1.
    norm_sq : float
        ||y||^2, the squared norm of the vector the quantized one is
        dotted against.
    d : int
        Dimension. Required.
    eps : float
        Relative accuracy in the tail statement.
    delta : float or None
        If given, the result also carries the smallest bit-width that
        attains (eps, delta) on this route and tail.
    x_norm_sq : float
        ||x||^2; the paper's statements take x on the unit sphere.
    n_blocks : int
        Independently rotated blocks the d coordinates are packed into.
    route : str
        Variance route: "table", "panter_dite", "qjl", "lower_bound".
    tail : str
        Tail inequality: "chebyshev" or "sub_gaussian".

    Returns
    -------
    RichResult
        variance, rmse, relative_error, delta_bound, constant,
        block_dims, plus the route, the tail and the assumption each
        carries. `bits_needed` is present when `delta` was given.

    References
    ----------
    Zandieh, Han, Daliri and Karbasi (2025) arXiv:2504.19874,
    Theorems 2 and 3, Lemma 4, and the b = 1..4 distortion table. The
    tail inequality is standard and is NOT from the paper.
    """
    if d is None:
        raise ValueError("d (dimension) is required")
    b = int(bits)
    dims = _blocks_of(d, n_blocks)
    v = variance_bound(b, d, norm_sq, x_norm_sq, n_blocks, route)
    p = tail_probability(v, eps, norm_sq, x_norm_sq, tail)
    scale = math.sqrt(x_norm_sq * norm_sq)
    payload = {
        "variance": v,
        "rmse": math.sqrt(v),
        "relative_error": math.sqrt(v) / scale if scale > 0.0 else 0.0,
        "delta_bound": p,
        "eps": float(eps),
        "constant": distortion_constant(b, dims[0], route),
        "bits": b,
        "d": int(d),
        "n_blocks": int(n_blocks),
        "block_dims": dims,
        "route": route,
        "tail": tail,
        "assumption": ("variance only" if tail == "chebyshev"
                       else "error sub-Gaussian with the stated variance "
                            "proxy -- an assumption, not a paper result"),
        "method": "TurboQuant inner-product distortion bound",
    }
    if delta is not None:
        payload["target_delta"] = float(delta)
        payload["bits_needed"] = bits_required(eps, delta, d, norm_sq,
                                               x_norm_sq, n_blocks,
                                               route, tail)
    return RichResult(payload=payload)


tqipb = turboquant_inner_product_distortion_bound


def cheatsheet():
    return ("tqipb: TurboQuant inner-product distortion bounds. "
            "routes " + ", ".join(ROUTES) + "; tails " + ", ".join(TAILS))
