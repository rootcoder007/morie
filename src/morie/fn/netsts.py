# morie.fn -- function file (rootcoder007/morie)
r"""LSTM time-series forecasting, and what the gates are for.

A plain recurrent network multiplies by the same weight matrix at every
step, so a gradient travelling back :math:`T` steps is scaled by that
matrix :math:`T` times: it vanishes or explodes. The LSTM's answer is a
cell state updated **additively**,

.. math::
   f_t &= \sigma(W_f[h_{t-1}, x_t] + b_f), \quad
   i_t = \sigma(W_i[\cdot] + b_i), \quad
   o_t = \sigma(W_o[\cdot] + b_o),\\
   c_t &= f_t \odot c_{t-1} + i_t \odot \tanh(W_c[\cdot] + b_c),
   \qquad h_t = o_t \odot \tanh(c_t),

so the path from :math:`c_{t-1}` to :math:`c_t` is multiplication by
:math:`f_t` alone. With the forget gate open the gradient passes
essentially unchanged -- the constant error carousel. The anchor
measures that directly: it propagates a signal through 200 steps with
the gate open and with it closed, and compares.

**The forget-gate bias should start positive, and it is not a detail.**
At zero initialisation :math:`f \approx 0.5`, so the cell halves at
every step and memory is gone in a dozen steps before training begins.
Initialising :math:`b_f` to 1 or 2 opens the gate, and the anchor
measures the retention horizon under each.

**Forecasting recursively compounds error, and that is the honest
comparison.** Feeding a one-step model its own prediction accumulates
error at every step; a direct model trained per horizon does not, at
the cost of one model per step. Both are here, and the anchor measures
the gap growing with horizon rather than stating a preference.

**Scaling matters more than architecture at this size.** :math:`\tanh`
saturates outside roughly :math:`[-2, 2]`, so an unscaled series with
values in the hundreds puts every gate hard against its rails and the
network stops learning. Standardisation is applied and inverted around
the forecast.

References
----------
Hochreiter, S. & Schmidhuber, J. (1997) "Long Short-Term Memory",
*Neural Computation* 9(8), 1735-1780,
doi:10.1162/neco.1997.9.8.1735. The cell, and the constant error
carousel argument.

Gers, F. A., Schmidhuber, J. & Cummins, F. (2000) "Learning to Forget:
Continual Prediction with LSTM", *Neural Computation* 12(10),
2451-2471, doi:10.1162/089976600300015015. The forget gate itself.

Jozefowicz, R., Zaremba, W. & Sutskever, I. (2015) "An Empirical
Exploration of Recurrent Network Architectures", *Proceedings of the
32nd International Conference on Machine Learning*, PMLR 37,
2342-2350. The positive forget-gate bias initialisation.

Hewamalage, H., Bergmeir, C. & Bandara, K. (2021) "Recurrent Neural
Networks for Time Series Forecasting: Current status and future
directions", *International Journal of Forecasting* 37(1), 388-427,
doi:10.1016/j.ijforecast.2020.06.008. Recursive versus direct
strategies, and preprocessing.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["lstm_cell", "lstm_run", "gradient_retention",
           "lstm_forecast", "standardize"]

_EPS = 1e-12


def standardize(y):
    r"""Centre and scale: tanh saturates outside about [-2, 2], so an
    unscaled series pins every gate against its rails."""
    yv = [float(v) for v in y]
    mu = sum(yv) / len(yv)
    sd = k.sd(yv) if len(yv) > 1 else 1.0
    sd = sd if sd > _EPS else 1.0
    return [(v - mu) / sd for v in yv], mu, sd


def lstm_cell(x, h, c, W, b, forget_bias=0.0):
    r"""One step. Gate order is i, f, g, o.

    ``forget_bias`` is added to the forget pre-activation. At zero the
    gate sits near 0.5 and the cell halves every step, so memory is
    gone in a dozen steps before training starts.
    """
    d = len(h)
    if len(c) != d:
        raise ValueError("netsts: hidden and cell sizes differ")
    inp = list(x) + list(h)
    if len(W) != len(inp):
        raise ValueError("netsts: W has %d rows for an input of %d"
                         % (len(W), len(inp)))
    if len(b) != 4 * d:
        raise ValueError("netsts: the bias needs 4*hidden = %d entries,"
                         " got %d" % (4 * d, len(b)))
    z = [sum(inp[i] * W[i][j] for i in range(len(inp))) + b[j]
         for j in range(4 * d)]
    i_g = [k.sigmoid(z[j]) for j in range(d)]
    f_g = [k.sigmoid(z[d + j] + forget_bias) for j in range(d)]
    g_g = [math.tanh(z[2 * d + j]) for j in range(d)]
    o_g = [k.sigmoid(z[3 * d + j]) for j in range(d)]
    cn = [f_g[j] * c[j] + i_g[j] * g_g[j] for j in range(d)]
    hn = [o_g[j] * math.tanh(cn[j]) for j in range(d)]
    return hn, cn, {"i": i_g, "f": f_g, "g": g_g, "o": o_g}


def lstm_run(X, W, b, hidden, forget_bias=0.0):
    """Run a sequence, returning every hidden state and gate."""
    d = int(hidden)
    h = [0.0] * d
    c = [0.0] * d
    hs, cs, gates = [], [], []
    for row in X:
        h, c, g = lstm_cell(list(row), h, c, W, b,
                            forget_bias=forget_bias)
        hs.append(list(h))
        cs.append(list(c))
        gates.append(g)
    return hs, cs, gates


def gradient_retention(forget_value, steps):
    r"""How much of a signal survives ``steps`` at a given forget gate.

    The cell path is multiplication by :math:`f` alone, so retention is
    :math:`f^{T}` exactly -- which is the constant error carousel
    stated as a number.
    """
    f = float(forget_value)
    if not 0.0 <= f <= 1.0:
        raise ValueError("netsts: the forget value must be in [0, 1], "
                         "got %r" % (forget_value,))
    return f ** int(steps)


def lstm_forecast(y, horizon, hidden=8, n_lags=4, strategy="recursive",
                  forget_bias=1.0, seed=0, ridge=1e-6):
    r"""Forecast with an LSTM read-out fitted by least squares.

    The recurrence supplies the features and a linear read-out is fitted
    on top, which keeps the module about the CELL rather than about an
    optimiser. ``strategy="direct"`` fits one read-out per horizon step
    instead of feeding predictions back.
    """
    if strategy not in ("recursive", "direct"):
        raise ValueError("netsts: strategy must be recursive or "
                         "direct, got %r" % (strategy,))
    yv = k.vec(y)
    n = len(yv)
    H = int(horizon)
    p = int(n_lags)
    if n < p + H + 4:
        raise ValueError("netsts: %d observations is too few for %d "
                         "lags and a horizon of %d" % (n, p, H))
    zs, mu, sd = standardize(yv)
    d = int(hidden)
    rng = np.random.default_rng(seed)
    W = [[rng.standard_normal() * 0.3 for _ in range(4 * d)]
         for _ in range(1 + d)]
    b = [0.0] * (4 * d)

    def features(seq):
        hs, _, _ = lstm_run([[v] for v in seq], W, b, d,
                            forget_bias=forget_bias)
        return hs[-1]

    if strategy == "recursive":
        Xf, yf = [], []
        for t in range(p, n):
            Xf.append([1.0] + features(zs[t - p:t]))
            yf.append(zs[t])
        beta = k.lstsq(Xf, yf, ridge)
        st = list(zs)
        out = []
        for _h in range(H):
            f = [1.0] + features(st[-p:])
            nxt = sum(f[a] * beta[a] for a in range(len(beta)))
            st.append(nxt)
            out.append(nxt)
        betas = [beta]
    else:
        out, betas = [], []
        for hstep in range(1, H + 1):
            Xf, yf = [], []
            for t in range(p, n - hstep + 1):
                Xf.append([1.0] + features(zs[t - p:t]))
                yf.append(zs[t + hstep - 1])
            bh = k.lstsq(Xf, yf, ridge)
            betas.append(bh)
            f = [1.0] + features(zs[-p:])
            out.append(sum(f[a] * bh[a] for a in range(len(bh))))
    fc = [v * sd + mu for v in out]
    return RichResult(payload={
        "estimate": fc, "forecast": fc, "strategy": strategy,
        "hidden": d, "n_lags": p, "forget_bias": float(forget_bias),
        "mean": mu, "sd": sd, "n_models": len(betas),
        "retention_10": gradient_retention(
            k.sigmoid(forget_bias), 10),
        "method": "LSTM forecaster, Hochreiter & Schmidhuber (1997) "
                  "cell with Gers, Schmidhuber & Cummins (2000) forget "
                  "gate",
    })


def cheatsheet():
    return ("netsts: c_t = f*c_{t-1} + i*g is ADDITIVE, so the cell "
            "path is multiplication by f alone and retention is f^T "
            "exactly -- the constant error carousel. Initialise the "
            "forget bias POSITIVE: at 0 the gate sits near 0.5 and "
            "memory halves every step. Standardise, or tanh saturates "
            "and nothing learns. Recursive forecasting compounds error "
            "with horizon; direct costs one model per step.")


# compact alias per ledger/NAMING.md
lstmforecast = lstm_forecast

# public names resolved by fn/_lazy_map.json
neural_ts_lstm = lstm_forecast
neuraltslstm = lstm_forecast
