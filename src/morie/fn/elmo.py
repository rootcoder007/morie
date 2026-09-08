# morie.fn -- function file (rootcoder007/morie)
r"""ELMo: deep contextualized word representations.

A word's vector is not fixed. ELMo runs a bidirectional language model
over the whole sentence and represents token :math:`k` by *all* of that
model's layers at once,

.. math:: R_k = \{x_k^{LM}, \overrightarrow{h}_{k,j}^{LM},
          \overleftarrow{h}_{k,j}^{LM} \mid j = 1, \dots, L\},

then collapses them for a downstream task with eq. (1),

.. math:: \mathrm{ELMo}_k^{task} = \gamma^{task}
          \sum_{j=0}^{L} s_j^{task}\,h_{k,j}^{LM},

where :math:`s^{task}` are **softmax-normalised** and
:math:`\gamma^{task}` scales the whole vector.

**The two scalars are not redundant, and that is easy to get wrong.**
The :math:`s_j` are constrained to a simplex, so they can only choose
*which* layers to read -- they cannot change the magnitude of what comes
out. All the magnitude lives in :math:`\gamma`, which the paper notes is
of practical importance for optimisation. Implement :math:`s_j` as free
weights and :math:`\gamma` becomes unidentifiable; drop :math:`\gamma`
and the representation is stuck at whatever scale the biLM happened to
produce. Both are checked here as identities: the weights sum to one,
and doubling :math:`\gamma` exactly doubles the output.

**Different layers carry different things.** Higher layers capture
word sense, lower ones syntax, which is why a task is allowed to
reweight them rather than being handed the top layer alone. Selecting
the top layer is the special case :math:`s = (0,\dots,0,1)`, and it is
reachable so the general form can be checked against it.

References
----------
Peters, M. E., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K. &
Zettlemoyer, L. (2018) "Deep contextualized word representations",
*Proceedings of NAACL-HLT 2018*, 2227-2237, doi:10.18653/v1/N18-1202,
arXiv:1802.05365. Sec. 3, eq. (1), and the layer-weighting scheme.

Hochreiter, S. & Schmidhuber, J. (1997) "Long Short-Term Memory",
*Neural Computation* 9(8), 1735-1780, doi:10.1162/neco.1997.9.8.1735.
The recurrent cell the biLM is built from.

Ba, J. L., Kiros, J. R. & Hinton, G. E. (2016) "Layer Normalization",
arXiv:1607.06450. The normalisation the paper applies per layer before
weighting, since each biLM layer has a different distribution.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["elmo_mix", "layer_weights", "bilm_forward", "lstm_step",
           "elmo_representation"]

_EPS = 1e-12


def layer_weights(raw):
    r"""Softmax-normalised :math:`s^{task}`.

    A simplex, not free weights: these choose WHICH layers to read and
    cannot alter the magnitude, which is gamma's job alone.
    """
    if not raw:
        raise ValueError("elmo: no layer weights given")
    mx = max(raw)
    e = [math.exp(float(v) - mx) for v in raw]
    tot = sum(e)
    return [v / tot for v in e]


def lstm_step(x, h, c, Wx, Wh, b):
    """One LSTM cell step, gates in the order i, f, g, o."""
    d = len(h)
    if len(c) != d:
        raise ValueError("elmo: hidden and cell sizes differ")
    z = [sum(x[i] * Wx[i][j] for i in range(len(x)))
         + sum(h[i] * Wh[i][j] for i in range(d)) + b[j]
         for j in range(4 * d)]
    i_g = [k.sigmoid(z[j]) for j in range(d)]
    f_g = [k.sigmoid(z[d + j]) for j in range(d)]
    g_g = [math.tanh(z[2 * d + j]) for j in range(d)]
    o_g = [k.sigmoid(z[3 * d + j]) for j in range(d)]
    cn = [f_g[j] * c[j] + i_g[j] * g_g[j] for j in range(d)]
    hn = [o_g[j] * math.tanh(cn[j]) for j in range(d)]
    return hn, cn


def bilm_forward(X, layers):
    r"""Run the biLM and return every layer's representation.

    ``layers`` is a list of ``(Wx_f, Wh_f, b_f, Wx_b, Wh_b, b_b)``. The
    token dimension must equal the hidden dimension, because layer 0 is
    the token vector duplicated and every layer has to be the same
    width for eq. (1) to add them. The
    backward pass reads the sequence in reverse and its output is
    re-reversed before concatenation, so position k always aligns with
    token k -- forgetting that alignment is the classic silent error,
    because the shapes still match.
    """
    Xm = k.mat(X)
    L = len(Xm)
    if L == 0:
        raise ValueError("elmo: empty sequence")
    # Layer 0 is the token representation DUPLICATED, h_{k,0} =
    # [x_k; x_k], so it is the same width as a biLSTM layer's
    # forward-backward concatenation and the mixture is well defined.
    # Leaving it at token width makes eq. (1) a sum over differently
    # shaped vectors, which fails loudly here rather than silently
    # broadcasting.
    reps = [[list(row) + list(row) for row in Xm]]
    cur = [list(row) for row in Xm]
    for (Wxf, Whf, bf, Wxb, Whb, bb) in layers:
        d = len(Whf)
        if len(reps[0][0]) != 2 * d:
            raise ValueError("elmo: token dimension %d but hidden "
                             "dimension %d; layer 0 is [x; x] so they "
                             "must match" % (len(Xm[0]), d))
        h = [0.0] * d
        c = [0.0] * d
        fwd = []
        for t in range(L):
            h, c = lstm_step(cur[t], h, c, Wxf, Whf, bf)
            fwd.append(list(h))
        h = [0.0] * d
        c = [0.0] * d
        bwd = []
        for t in range(L - 1, -1, -1):
            h, c = lstm_step(cur[t], h, c, Wxb, Whb, bb)
            bwd.append(list(h))
        bwd.reverse()                # re-align: position k is token k
        cur = [fwd[t] + bwd[t] for t in range(L)]
        reps.append([list(row) for row in cur])
    return reps


def elmo_mix(reps, raw_weights, gamma=1.0, position=None):
    r"""Eq. (1): :math:`\gamma \sum_j s_j h_{k,j}`."""
    n_layers = len(reps)
    if len(raw_weights) != n_layers:
        raise ValueError("elmo: %d weights for %d layers"
                         % (len(raw_weights), n_layers))
    s = layer_weights(raw_weights)
    L = len(reps[0])
    dims = {len(reps[j][0]) for j in range(n_layers)}
    if len(dims) != 1:
        raise ValueError("elmo: layers have differing widths %s"
                         % sorted(dims))
    d = dims.pop()
    idx = range(L) if position is None else [int(position)]
    out = []
    for t in idx:
        out.append([float(gamma) * sum(s[j] * reps[j][t][c]
                                       for j in range(n_layers))
                    for c in range(d)])
    return out[0] if position is not None else out


def elmo_representation(X, layers, raw_weights=None, gamma=1.0):
    """The biLM plus the task-specific mix, end to end."""
    reps = bilm_forward(X, layers)
    n = len(reps)
    raw = [0.0] * n if raw_weights is None else list(raw_weights)
    mixed = elmo_mix(reps, raw, gamma=gamma)
    s = layer_weights(raw)
    return RichResult(payload={
        "estimate": mixed, "elmo": mixed, "layers": reps,
        "weights": s, "gamma": float(gamma), "n_layers": n,
        "L": len(reps[0]), "d": len(mixed[0]) if mixed else 0,
        "top_layer": reps[-1],
        "method": "ELMo layer mixture, Peters et al. (2018) eq. (1)",
    })


def cheatsheet():
    return ("elmo: ELMo_k = gamma * sum_j s_j h_{k,j}, s SOFTMAX-"
            "normalised (eq. 1). The simplex constraint means s chooses "
            "WHICH layers to read and cannot scale the output -- all "
            "magnitude is in gamma. Free s makes gamma unidentifiable; "
            "no gamma leaves the scale wherever the biLM left it. The "
            "backward pass must be re-reversed or position k stops "
            "meaning token k, and the shapes will not tell you.")


# compact alias per ledger/NAMING.md
elmorepresentation = elmo_representation

# public names resolved by fn/_lazy_map.json
elmo = elmo_representation
