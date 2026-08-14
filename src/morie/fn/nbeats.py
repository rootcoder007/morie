# morie.fn -- function file (rootcoder007/morie)
r"""N-BEATS: doubly residual stacking for forecasting.

A block reads a lookback window and emits two things: a *forecast* of
the horizon and a *backcast*, its own reconstruction of the window it
just read. The backcast is what makes the architecture work.

**Doubly residual stacking.** Each block subtracts what it explained
from the input before the next block sees it, and the forecasts add up:

.. math:: x_\ell = x_{\ell-1} - \hat x_{\ell-1}, \qquad
          \hat y = \sum_\ell \hat y_\ell.

So block :math:`\ell` is fitted to the part of the signal its
predecessors could not represent. The residual telescopes exactly:
:math:`x_L = x_0 - \sum_{\ell<L}\hat x_\ell`, which the anchor checks
as an identity rather than an intention -- feed the residual forward
without subtracting and every block re-explains the same trend, which
still trains and simply wastes the stack.

**Interpretability comes from constraining the basis, not from
inspecting the weights.** A generic block lets :math:`\theta` be an
arbitrary linear map. A *trend* block forces the output to be a
low-order polynomial in normalised time, and a *seasonality* block
forces it onto a Fourier basis. Those are hard constraints on the
function space, so the trend stack's output really is a polynomial --
checkable by fitting one and comparing residuals -- rather than merely
being labelled "trend".

References
----------
Oreshkin, B. N., Carpov, D., Chapados, N. & Bengio, Y. (2020)
"N-BEATS: Neural basis expansion analysis for interpretable time series
forecasting", *International Conference on Learning Representations*,
arXiv:1905.10437. Sec. 3, the doubly residual topology, and the trend
and seasonality bases.

Challu, C., Olivares, K. G., Oreshkin, B. N., Garza, F., Mergenthaler-
Canseco, M. & Dubrawski, A. (2023) "NHITS: Neural Hierarchical
Interpolation for Time Series Forecasting", *Proceedings of the AAAI
Conference on Artificial Intelligence* 37(6), 6989-6997,
doi:10.1609/aaai.v37i6.25854, arXiv:2201.12886. The successor that adds
multi-rate sampling.

Hyndman, R. J. & Athanasopoulos, G. (2021) *Forecasting: Principles and
Practice*, 3rd edn, OTexts. The decomposition vocabulary the
interpretable blocks borrow.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["trend_basis", "seasonality_basis", "nbeats_block",
           "nbeats_stack", "nbeats_forecast"]

_EPS = 1e-12


def trend_basis(length, degree, offset=0.0, scale=None):
    r"""Powers of normalised time: the polynomial the trend block is
    constrained to."""
    if degree < 0:
        raise ValueError("nbeats: degree must be non-negative, got %d"
                         % degree)
    sc = float(length) if scale is None else float(scale)
    return [[((offset + t) / sc) ** p for t in range(length)]
            for p in range(degree + 1)]


def seasonality_basis(length, harmonics, offset=0.0, period=None):
    r"""Cosine and sine pairs: the Fourier basis of the seasonality
    block."""
    if harmonics < 1:
        raise ValueError("nbeats: need at least 1 harmonic, got %d"
                         % harmonics)
    per = float(length) if period is None else float(period)
    rows = []
    for h in range(1, int(harmonics) + 1):
        rows.append([math.cos(2.0 * math.pi * h * (offset + t) / per)
                     for t in range(length)])
        rows.append([math.sin(2.0 * math.pi * h * (offset + t) / per)
                     for t in range(length)])
    return rows


def _fit_theta(y, basis, ridge=1e-8):
    """Least squares of y on the basis rows."""
    L = len(y)
    X = [[basis[p][t] for p in range(len(basis))] for t in range(L)]
    return k.lstsq(X, list(y), ridge)


def nbeats_block(window, horizon, kind="generic", degree=2,
                 harmonics=3, ridge=1e-8):
    r"""One block: fit the basis to the window, emit backcast and
    forecast.

    The SAME theta drives both, evaluated on the backcast basis over
    the lookback and the forecast basis over the horizon -- which is
    what ties the two outputs together and makes the residual
    meaningful.
    """
    if kind not in ("generic", "trend", "seasonality"):
        raise ValueError("nbeats: kind must be generic, trend or "
                         "seasonality, got %r" % (kind,))
    L = len(window)
    H = int(horizon)
    if H < 1:
        raise ValueError("nbeats: horizon must be at least 1, got %d"
                         % H)
    if kind == "trend":
        bb = trend_basis(L, degree, scale=L)
        fb = trend_basis(H, degree, offset=L, scale=L)
    elif kind == "seasonality":
        bb = seasonality_basis(L, harmonics, period=L)
        fb = seasonality_basis(H, harmonics, offset=L, period=L)
    else:
        # generic: an unconstrained basis, here the identity over the
        # lookback and a free constant-plus-slope over the horizon
        bb = [[1.0 if t == p else 0.0 for t in range(L)]
              for p in range(L)]
        fb = [[1.0 / max(L, 1)] * H for _ in range(L)]
    theta = _fit_theta(window, bb, ridge)
    backcast = [sum(theta[p] * bb[p][t] for p in range(len(bb)))
                for t in range(L)]
    forecast = [sum(theta[p] * fb[p][t] for p in range(len(fb)))
                for t in range(H)]
    return backcast, forecast, theta


def nbeats_stack(window, horizon, blocks, ridge=1e-8):
    r"""Doubly residual stacking, eq. of Sec. 3.2.

    ``blocks`` is a list of ``(kind, degree, harmonics)``.
    """
    resid = [float(v) for v in window]
    total = [0.0] * int(horizon)
    trace = []
    for (kind, deg, harm) in blocks:
        bc, fc, th = nbeats_block(resid, horizon, kind=kind,
                                  degree=deg, harmonics=harm,
                                  ridge=ridge)
        resid = [resid[t] - bc[t] for t in range(len(resid))]
        total = [total[h] + fc[h] for h in range(len(total))]
        trace.append({"kind": kind, "backcast": bc, "forecast": fc,
                      "theta": th,
                      "residual_norm": math.sqrt(sum(v * v
                                                     for v in resid))})
    return total, resid, trace


def nbeats_forecast(y, horizon, lookback=None, blocks=None, ridge=1e-8):
    r"""Forecast from the last ``lookback`` observations."""
    yv = k.vec(y)
    n = len(yv)
    H = int(horizon)
    lb = min(n, int(lookback) if lookback else min(n, max(8, 3 * H)))
    if lb < 4:
        raise ValueError("nbeats: lookback of %d is too short" % lb)
    if n < lb:
        raise ValueError("nbeats: %d observations for a lookback of %d"
                         % (n, lb))
    blk = ([("trend", 2, 3), ("seasonality", 2, 3), ("trend", 1, 3)]
           if blocks is None else list(blocks))
    window = yv[n - lb:]
    fc, resid, trace = nbeats_stack(window, H, blk, ridge=ridge)
    explained = [window[t] - resid[t] for t in range(lb)]
    return RichResult(payload={
        "estimate": fc, "forecast": fc, "residual": resid,
        "backcast": explained, "blocks": trace, "lookback": lb,
        "horizon": H, "n": n,
        "residual_norm": math.sqrt(sum(v * v for v in resid)),
        "window_norm": math.sqrt(sum(v * v for v in window)),
        "n_blocks": len(blk),
        "method": "N-BEATS doubly residual stacking, Oreshkin, Carpov, "
                  "Chapados & Bengio (2020)",
    })


def cheatsheet():
    return ("nbeats: each block emits a BACKCAST and a forecast from "
            "one theta. Residual in: x_l = x_{l-1} - xhat_{l-1}; "
            "forecasts out: yhat = sum_l yhat_l. The residual "
            "telescopes exactly, so block l only ever sees what its "
            "predecessors could not explain -- skip the subtraction and "
            "every block re-fits the same trend. Trend and seasonality "
            "blocks CONSTRAIN the basis (polynomial, Fourier); that is "
            "where interpretability comes from.")


# compact alias per ledger/NAMING.md
nbeatsforecast = nbeats_forecast

# public names resolved by fn/_lazy_map.json
n_beats = nbeats_forecast
nbeats = nbeats_forecast
