# morie.fn -- function file (rootcoder007/morie)
r"""Mamba's selective state space step (S6).

A structured state space model runs the linear recurrence

.. math:: h_t = \bar A h_{t-1} + \bar B x_t, \qquad y_t = C h_t,

with :math:`(\bar A, \bar B)` obtained from continuous
:math:`(\Delta, A, B)` by a discretisation rule -- zero-order hold:

.. math:: \bar A = \exp(\Delta A), \qquad
          \bar B = (\Delta A)^{-1}(\exp(\Delta A) - I)\cdot \Delta B.

**What makes S6 selective is that these stop being constants.** In S4
the parameters are fixed for every timestep, which makes the model
linear time invariant and lets it be run as a convolution. Mamba makes
:math:`B`, :math:`C` and :math:`\Delta` functions of the input,

.. math:: s_B(x) = \mathrm{Linear}_N(x), \quad
          s_C(x) = \mathrm{Linear}_N(x), \quad
          s_\Delta(x) = \mathrm{Broadcast}_D(\mathrm{Linear}_1(x)),
          \quad \tau_\Delta = \mathrm{softplus},

so the dynamics vary with content. That is what lets the model ignore a
token: the price is that the convolution is gone and only the recurrent
scan remains.

**Why Delta is projected to one dimension and broadcast.** If a token is
to be ignored it must be ignored by *every* channel, so
:math:`s_\Delta` maps the input to a single number before it is repeated
across the D channels. A per-channel :math:`\Delta` would let some
channels keep a token the others dropped, which is not the mechanism the
paper describes.

**The identity that pins the implementation down.** Theorem 1: with
:math:`N=1`, :math:`A=-1`, :math:`B=1`, :math:`s_\Delta` linear and
:math:`\tau_\Delta` softplus, the recurrence is *exactly* a gated RNN,

.. math:: g_t = \sigma(\mathrm{Linear}(x_t)), \qquad
          h_t = (1 - g_t)h_{t-1} + g_t x_t.

The algebra is worth keeping in view because it is what the anchor
checks to machine precision: with :math:`\Delta = \mathrm{softplus}(s)`,
:math:`\bar A = \exp(-\Delta) = 1/(1+e^{s}) = 1 - \sigma(s)` and
:math:`\bar B = 1 - \exp(-\Delta) = \sigma(s)`. A ZOH written with the
common simplification :math:`\bar B \approx \Delta B` does *not* satisfy
it -- the identity fails by a visible margin -- which is exactly the
kind of substitution that otherwise passes unnoticed.

References
----------
Gu, A. & Dao, T. (2023) "Mamba: Linear-Time Sequence Modeling with
Selective State Spaces", arXiv:2312.00752. Sec. 2 (eq. 1-4, the ZOH
rule), Sec. 3.2 (Algorithm 2, S6, and the choices of s_B, s_C, s_Delta,
tau_Delta), Theorem 1 of Sec. 3.5.1.

Gu, A., Goel, K. & Re, C. (2022) "Efficiently Modeling Long Sequences
with Structured State Spaces", *International Conference on Learning
Representations*, arXiv:2111.00396. S4, the time-invariant model
Algorithm 2 modifies.

Gu, A., Gupta, A., Goel, K. & Re, C. (2022) "On the Parameterization and
Initialization of Diagonal State Space Models", *Advances in Neural
Information Processing Systems* 35, arXiv:2206.11893. The diagonal
structure for A used here.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["softplus", "discretize_zoh", "selective_ssm_step",
           "selective_scan", "gated_rnn_equivalent", "s6_layer"]

_EPS = 1e-12


def softplus(z):
    """log(1 + exp(z)), computed without overflowing for large z."""
    x = float(z)
    if x > 30.0:
        return x
    if x < -30.0:
        return math.exp(x)
    return math.log1p(math.exp(x))


def discretize_zoh(delta, A, B, rule="zoh"):
    r"""(Delta, A, B) -> (Abar, Bbar), eq. (4), for diagonal A.

    ``rule="zoh"`` is the paper's zero-order hold,
    :math:`\bar A = \exp(\Delta A)` and
    :math:`\bar B = (\Delta A)^{-1}(\exp(\Delta A) - I)\Delta B`, which
    for a diagonal A is elementwise
    :math:`(\exp(\Delta A_n) - 1)B_n / A_n`.

    ``rule="euler"`` is the common simplification
    :math:`\bar B = \Delta B`. It is offered because implementations do
    use it, and refused as a silent default because it breaks Theorem 1.
    """
    if rule not in ("zoh", "euler"):
        raise ValueError("mambss: rule must be zoh or euler, got %r"
                         % (rule,))
    d = float(delta)
    if d < 0.0:
        raise ValueError("mambss: delta must be non-negative, got %r"
                         % (delta,))
    Av = [float(v) for v in A]
    Bv = [float(v) for v in B]
    if len(Av) != len(Bv):
        raise ValueError("mambss: A has %d entries but B has %d"
                         % (len(Av), len(Bv)))
    Abar, Bbar = [], []
    for n in range(len(Av)):
        da = d * Av[n]
        ea = math.exp(da)
        Abar.append(ea)
        if rule == "euler":
            Bbar.append(d * Bv[n])
        elif abs(da) < 1e-8:
            # (exp(da) - 1)/A -> d as A -> 0; use the series so the
            # limit is smooth rather than a 0/0
            Bbar.append(d * Bv[n] * (1.0 + 0.5 * da))
        else:
            Bbar.append((ea - 1.0) / Av[n] * Bv[n])
    return Abar, Bbar


def selective_ssm_step(x, h, A, B, C, delta, rule="zoh"):
    r"""One timestep of the selective recurrence, for one channel.

    ``h`` is the length-N state, ``x`` the scalar input for this
    channel, and ``B``, ``C``, ``delta`` are this timestep's values --
    which is what makes it selective rather than time invariant.
    """
    N = len(A)
    if len(h) != N:
        raise ValueError("mambss: state has %d entries but A has %d"
                         % (len(h), N))
    if len(C) != N:
        raise ValueError("mambss: C has %d entries but A has %d"
                         % (len(C), N))
    Abar, Bbar = discretize_zoh(delta, A, B, rule=rule)
    hn = [Abar[n] * h[n] + Bbar[n] * float(x) for n in range(N)]
    y = sum(C[n] * hn[n] for n in range(N))
    return hn, y


def _linear(x, Wm, b):
    """Wm @ x + b, with Wm given as rows."""
    return [sum(Wm[r][c] * x[c] for c in range(len(x))) + b[r]
            for r in range(len(Wm))]


def selective_scan(X, A, W_B, W_C, W_delta, delta_bias=None,
                   b_B=None, b_C=None, b_delta=0.0, rule="zoh",
                   D_skip=None):
    r"""Algorithm 2: run S6 over a sequence.

    Parameters
    ----------
    X : array-like
        The input sequence, L rows of D channels.
    A : array-like
        The diagonal state matrix, D rows of N entries. A parameter, not
        a function of the input.
    W_B, W_C : array-like
        Projections giving :math:`s_B(x), s_C(x) \in \mathbb{R}^N` at
        each step: N rows of D.
    W_delta : array-like
        One row of D. Note the singular row: :math:`s_\Delta` maps to
        ONE dimension and is broadcast across channels, so a token
        ignored by one channel is ignored by all.
    delta_bias : array-like, optional
        The per-channel Parameter added before the softplus, length D.
    D_skip : array-like, optional
        The usual per-channel skip connection, y += D * x.

    Returns
    -------
    RichResult
        ``y`` is the L-by-D output, with the per-step ``delta`` and the
        final state.
    """
    Xm = k.mat(X)
    L = len(Xm)
    if L == 0:
        raise ValueError("mambss: the input sequence is empty")
    D = len(Xm[0])
    Am = k.mat(A)
    if len(Am) != D:
        raise ValueError("mambss: A has %d rows for %d channels"
                         % (len(Am), D))
    N = len(Am[0])
    WB, WC = k.mat(W_B), k.mat(W_C)
    if len(WB) != N or len(WC) != N:
        raise ValueError("mambss: W_B and W_C must have N=%d rows, got "
                         "%d and %d" % (N, len(WB), len(WC)))
    Wd = k.mat(W_delta)
    if len(Wd) != 1:
        raise ValueError("mambss: W_delta must have exactly 1 row -- "
                         "s_Delta projects to one dimension and is "
                         "broadcast over channels; got %d" % len(Wd))
    bB = [0.0] * N if b_B is None else [float(v) for v in b_B]
    bC = [0.0] * N if b_C is None else [float(v) for v in b_C]
    dbias = ([0.0] * D if delta_bias is None
             else [float(v) for v in delta_bias])
    if len(dbias) != D:
        raise ValueError("mambss: delta_bias has %d entries for %d "
                         "channels" % (len(dbias), D))
    skip = ([0.0] * D if D_skip is None else [float(v) for v in D_skip])

    h = [[0.0] * N for _ in range(D)]
    Y, deltas = [], []
    for t in range(L):
        xt = [float(v) for v in Xm[t]]
        Bt = _linear(xt, WB, bB)
        Ct = _linear(xt, WC, bC)
        raw = _linear(xt, Wd, [float(b_delta)])[0]
        dt = [softplus(raw + dbias[c]) for c in range(D)]
        deltas.append(dt)
        row = []
        for c in range(D):
            h[c], yc = selective_ssm_step(xt[c], h[c], Am[c], Bt, Ct,
                                          dt[c], rule=rule)
            row.append(yc + skip[c] * xt[c])
        Y.append(row)
    return RichResult(payload={
        "y": Y, "estimate": Y, "state": h, "delta": deltas,
        "L": L, "D": D, "N": N, "rule": rule,
        "time_invariant": False,
        "method": "selective state space scan (S6), Gu & Dao (2023) "
                  "Algorithm 2",
    })


def gated_rnn_equivalent(x, w, b=0.0):
    r"""Theorem 1's right-hand side, for checking against the scan.

    :math:`g_t = \sigma(w x_t + b)`,
    :math:`h_t = (1-g_t)h_{t-1} + g_t x_t`.
    """
    h = 0.0
    hs, gs = [], []
    for v in x:
        g = k.sigmoid(float(w) * float(v) + float(b))
        h = (1.0 - g) * h + g * float(v)
        hs.append(h)
        gs.append(g)
    return hs, gs


def s6_layer(X, A, W_B, W_C, W_delta, **kw):
    """Convenience wrapper returning just the output sequence."""
    return selective_scan(X, A, W_B, W_C, W_delta, **kw)["y"]


def cheatsheet():
    return ("mambss: S6. B, C, Delta are FUNCTIONS of x (Alg. 2), so "
            "the model is time-varying and only the scan works -- no "
            "convolution. ZOH: Abar = exp(Delta A), Bbar = "
            "(exp(Delta A) - 1) B / A. s_Delta projects to ONE dim then "
            "broadcasts over D. Theorem 1: N=1, A=-1, B=1, softplus "
            "gives exactly g = sigmoid(Linear(x)), h = (1-g)h + g x.")


# compact alias per ledger/NAMING.md
selectivessmstep = selective_ssm_step
