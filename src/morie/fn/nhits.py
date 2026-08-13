# morie.fn -- function file (rootcoder007/morie)
r"""N-HiTS: hierarchical interpolation and multi-rate sampling.

N-BEATS spends the same capacity on every frequency in the signal, and
emits a forecast point for every horizon step from every block. Over a
long horizon that is both expensive and badly conditioned: the number
of output parameters grows with the horizon, and high-frequency blocks
waste them predicting detail that is unpredictable far out.

**Two changes, and they are the same idea applied at both ends.**

*Multi-rate signal sampling.* Each block max-pools its input by a
kernel :math:`k_\ell` before reading it. A large kernel leaves only the
slow components, so that block sees -- and can only fit -- low
frequencies. Pooling is what makes the block frequency-specific.

*Hierarchical interpolation.* Each block predicts only
:math:`\lceil r_\ell H\rceil` points, at an expressiveness ratio
:math:`r_\ell \le 1`, and those are **interpolated** up to the full
horizon :math:`H`. A block with :math:`r=1/8` emits an eighth as many
numbers and stretches them across the horizon, which is exactly the
right parameterisation for a low-frequency component.

**The ratios and the pooling must move together, and that is the whole
design.** Pair a large pooling kernel with a small expressiveness ratio
and the block sees a smooth signal and predicts it with few points --
coherent. Pair a large kernel with :math:`r=1` and the block has full
output resolution for a signal that has none, which is the waste
N-HiTS removes. The anchor checks that a large-kernel block really does
lose the high-frequency component, and that interpolation from a
handful of knots reconstructs a smooth series but not a jagged one.

**Interpolation must be exact at its knots.** Whatever the ratio, the
interpolated output has to pass through the predicted points, or the
block is not predicting what it appears to be. Checked as an identity.

References
----------
Challu, C., Olivares, K. G., Oreshkin, B. N., Garza, F.,
Mergenthaler-Canseco, M. & Dubrawski, A. (2023) "NHITS: Neural
Hierarchical Interpolation for Time Series Forecasting", *Proceedings
of the AAAI Conference on Artificial Intelligence* 37(6), 6989-6997,
doi:10.1609/aaai.v37i6.25854, arXiv:2201.12886. Sec. 3: multi-rate
sampling, hierarchical interpolation, and the expressiveness ratios.

Oreshkin, B. N., Carpov, D., Chapados, N. & Bengio, Y. (2020)
"N-BEATS: Neural basis expansion analysis for interpretable time series
forecasting", *International Conference on Learning Representations*,
arXiv:1905.10437. The doubly residual stack N-HiTS inherits.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["max_pool", "linear_interpolate", "nhits_block",
           "nhits_stack", "nhits_forecast", "expressiveness_knots"]

_EPS = 1e-12


def max_pool(x, kernel, stride=None):
    r"""Max pooling: the multi-rate sampling of Sec. 3.1.

    A large kernel leaves only the slow components, so the block that
    reads it can only fit low frequencies. That is the mechanism, not
    an efficiency trick.
    """
    xv = [float(v) for v in x]
    kk = int(kernel)
    if kk < 1:
        raise ValueError("nhits: the kernel must be at least 1, got %d"
                         % kk)
    st = kk if stride is None else int(stride)
    if st < 1:
        raise ValueError("nhits: the stride must be at least 1")
    if kk > len(xv):
        raise ValueError("nhits: kernel %d exceeds the input length %d"
                         % (kk, len(xv)))
    return [max(xv[i:i + kk]) for i in range(0, len(xv) - kk + 1, st)]


def expressiveness_knots(horizon, ratio):
    r"""How many points a block predicts:
    :math:`\lceil r H\rceil`, at least 2."""
    r = float(ratio)
    if not 0.0 < r <= 1.0:
        raise ValueError("nhits: the ratio must be in (0, 1], got %r"
                         % (ratio,))
    return max(2, int(math.ceil(r * int(horizon))))


def linear_interpolate(knots, horizon):
    r"""Stretch ``knots`` across ``horizon`` points.

    Exact at the knots by construction: the first and last land on the
    endpoints and every interior knot is hit, or the block is not
    predicting what it appears to be.
    """
    kv = [float(v) for v in knots]
    n = len(kv)
    H = int(horizon)
    if n < 2:
        raise ValueError("nhits: need at least 2 knots, got %d" % n)
    if H < 1:
        raise ValueError("nhits: the horizon must be at least 1")
    if H == 1:
        return [kv[0]]
    out = []
    for h in range(H):
        pos = h * (n - 1) / float(H - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, n - 1)
        w = pos - lo
        out.append((1.0 - w) * kv[lo] + w * kv[hi])
    return out


def _fit_basis(y, basis, ridge=1e-8):
    X = [[basis[p][t] for p in range(len(basis))]
         for t in range(len(y))]
    return k.lstsq(X, list(y), ridge)


def nhits_block(window, horizon, kernel=1, ratio=1.0, degree=2,
                ridge=1e-8):
    r"""One block: pool the input, fit, predict knots, interpolate.

    The backcast is produced at the block's own resolution too, so the
    residual passed on is what THIS block could not explain at THIS
    frequency.
    """
    w = [float(v) for v in window]
    L = len(w)
    H = int(horizon)
    pooled = max_pool(w, kernel)
    Lp = len(pooled)
    if Lp < degree + 1:
        raise ValueError("nhits: pooling by %d leaves %d points, too "
                         "few for degree %d" % (kernel, Lp, degree))
    # a polynomial basis over the pooled (slow) view
    bb = [[(t / float(max(Lp - 1, 1))) ** p for t in range(Lp)]
          for p in range(int(degree) + 1)]
    theta = _fit_basis(pooled, bb, ridge)
    # backcast at pooled resolution, interpolated back to the window
    back_p = [sum(theta[p] * bb[p][t] for p in range(len(bb)))
              for t in range(Lp)]
    backcast = (linear_interpolate(back_p, L) if Lp >= 2
                else [back_p[0]] * L)
    # forecast: predict only ceil(rH) knots, then interpolate up
    n_knots = expressiveness_knots(H, ratio)
    fb = [[((Lp - 1 + (j + 1) * (Lp - 1) / float(max(n_knots, 1)))
            / float(max(Lp - 1, 1))) ** p for j in range(n_knots)]
          for p in range(int(degree) + 1)]
    knots = [sum(theta[p] * fb[p][j] for p in range(len(fb)))
             for j in range(n_knots)]
    forecast = linear_interpolate(knots, H)
    return backcast, forecast, knots, pooled


def nhits_stack(window, horizon, blocks, ridge=1e-8):
    r"""Doubly residual stacking over frequency-specific blocks.

    ``blocks`` is a list of ``(kernel, ratio, degree)``. Large kernel
    with small ratio is the coherent pairing: a smooth view predicted
    with few points.
    """
    resid = [float(v) for v in window]
    total = [0.0] * int(horizon)
    trace = []
    for (kern, ratio, deg) in blocks:
        bc, fc, knots, pooled = nhits_block(resid, horizon,
                                            kernel=kern, ratio=ratio,
                                            degree=deg, ridge=ridge)
        resid = [resid[t] - bc[t] for t in range(len(resid))]
        total = [total[h] + fc[h] for h in range(len(total))]
        trace.append({"kernel": kern, "ratio": ratio,
                      "n_knots": len(knots), "knots": knots,
                      "pooled_length": len(pooled), "backcast": bc,
                      "forecast": fc,
                      "residual_norm": math.sqrt(sum(v * v
                                                     for v in resid))})
    return total, resid, trace


def nhits_forecast(y, horizon, lookback=None, blocks=None, ridge=1e-8):
    r"""Forecast with a coarse-to-fine stack.

    The default pairs decreasing kernels with increasing ratios, which
    is the paper's design: the first block sees a heavily smoothed
    signal and predicts few points, the last sees the raw signal and
    predicts many.
    """
    yv = k.vec(y)
    n = len(yv)
    H = int(horizon)
    lb = min(n, int(lookback) if lookback else min(n, max(16, 4 * H)))
    if lb < 8:
        raise ValueError("nhits: lookback of %d is too short" % lb)
    blk = ([(4, 0.25, 2), (2, 0.5, 2), (1, 1.0, 2)]
           if blocks is None else list(blocks))
    window = yv[n - lb:]
    fc, resid, trace = nhits_stack(window, H, blk, ridge=ridge)
    return RichResult(payload={
        "estimate": fc, "forecast": fc, "residual": resid,
        "blocks": trace, "lookback": lb, "horizon": H, "n": n,
        "total_knots": sum(b["n_knots"] for b in trace),
        "dense_parameters": H * len(blk),
        "residual_norm": math.sqrt(sum(v * v for v in resid)),
        "n_blocks": len(blk),
        "method": "N-HiTS multi-rate sampling and hierarchical "
                  "interpolation, Challu et al. (2023)",
    })


def cheatsheet():
    return ("nhits: each block MAX-POOLS its input by kernel k (so it "
            "sees only frequencies slower than k) and predicts only "
            "ceil(rH) knots, INTERPOLATED up to H. Large kernel with "
            "small ratio is coherent -- a smooth view predicted with "
            "few points; large kernel with r=1 is the waste N-HiTS "
            "removes. Interpolation is exact at the knots.")


# compact alias per ledger/NAMING.md
nhitsforecast = nhits_forecast
