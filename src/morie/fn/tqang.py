r"""Uniform angle quantisation on :math:`[-\pi, \pi)`.

Splits the circle into :math:`2^b` equal sectors and represents each
angle by its sector's midpoint. With :math:`\Delta = 2\pi/2^b` the
codewords are

.. math:: \hat\theta_k = -\pi + (k + \tfrac12)\Delta,
          \qquad k = 0, \ldots, 2^b - 1,

so the worst-case error is :math:`\Delta/2` and, for angles uniform on
the circle, the mean squared error is :math:`\Delta^2/12` -- the flat
quantiser result, which holds here exactly because the sectors are
equal and the density is flat.

The one thing that makes angles different from ordinary scalars is
**wraparound**: an angle just below :math:`\pi` and one just above
:math:`-\pi` are neighbours, not opposites. Every operation here works
on the wrapped difference

.. math:: d(\alpha, \beta) = ((\alpha - \beta + \pi) \bmod 2\pi) - \pi

so the reported error is the true angular distance. A quantiser that
used the naive difference would report an error of nearly
:math:`2\pi` at the seam while the actual error was near zero.
"""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["quantize_angles", "wrap_angle", "angular_difference", "tqang"]

TWO_PI = 2.0 * math.pi


def wrap_angle(theta):
    """Wrap to [-pi, pi)."""
    t = math.fmod(float(theta) + math.pi, TWO_PI)
    if t < 0.0:
        t += TWO_PI
    return t - math.pi


def angular_difference(a, b):
    """Signed shortest difference a - b, in [-pi, pi)."""
    return wrap_angle(float(a) - float(b))


def quantize_angles(theta, bits=4):
    r"""Quantise angles to :math:`2^{\text{bits}}` uniform sectors.

    Returns
    -------
    RichResult
        ``indices`` are sector numbers, ``values`` the reconstructed
        angles, ``errors`` the WRAPPED errors, and ``mse`` their mean
        square. ``mse_bound`` is :math:`\Delta^2/12`, the flat-density
        value the empirical MSE should approach.
    """
    b = int(bits)
    if not (1 <= b <= 30):
        raise ValueError("quantize_angles: bits must lie in 1..30, got %r"
                         % (bits,))
    n_levels = 1 << b
    delta = TWO_PI / n_levels

    idx, val, err = [], [], []
    for t in np.atleast_1d(np.asarray(theta, dtype=float)):
        w = wrap_angle(float(t))
        k = int(math.floor((w + math.pi) / delta))
        # An angle exactly at +pi wraps to -pi, but floating point can
        # still land k at n_levels; clamp rather than index out.
        if k >= n_levels:
            k = n_levels - 1
        if k < 0:
            k = 0
        rec = -math.pi + (k + 0.5) * delta
        idx.append(k)
        val.append(rec)
        err.append(angular_difference(w, rec))

    mse = sum(e * e for e in err) / float(len(err)) if err else 0.0
    return RichResult(payload={
        "estimate": val,
        "indices": idx,
        "values": val,
        "errors": err,
        "mse": float(mse),
        "max_abs_error": float(max((abs(e) for e in err), default=0.0)),
        "delta": delta,
        "half_delta": 0.5 * delta,
        "mse_bound": delta * delta / 12.0,
        "bits": b,
        "levels": n_levels,
        "method": "Uniform angle quantisation on [-pi, pi), midpoint "
                  "reconstruction, wrapped error",
    })


def cheatsheet():
    return ("tqang: 2^b equal sectors, delta = 2pi/2^b, codeword "
            "-pi + (k+0.5) delta; |err| <= delta/2, MSE -> delta^2/12; "
            "all errors use the WRAPPED difference.")


tqang = quantize_angles

# public names resolved by fn/_lazy_map.json
turboquant_angle_quantization = quantize_angles
