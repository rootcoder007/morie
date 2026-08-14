# morie.fn -- function file (rootcoder007/morie)
r"""xDeepFM: explicit, vector-wise, bounded-degree interactions.

A plain deep network can approximate any function, so it will
eventually represent feature interactions -- but *implicitly*, and at
the **bit-wise** level: the units mix individual embedding
*coordinates* across fields. Factorization machines model interactions
at the **vector-wise** level, where a whole field embedding interacts
with another whole field embedding. Those are different objects, and
the paper's argument is that the difference matters and that a DNN's
implicit interactions leave the maximum degree unknown.

**The Compressed Interaction Network.** CIN builds interactions
explicitly, one degree per layer:

.. math:: X^{k}_{h,*} = \sum_{i=1}^{m_{k-1}}\sum_{j=1}^{m}
          W^{k,h}_{ij}\,\big(X^{k-1}_{i,*} \circ X^0_{j,*}\big),

where :math:`\circ` is the element-wise (Hadamard) product. Layer
:math:`k` therefore contains exactly degree-:math:`(k+1)` interactions
-- the degree is *bounded and known*, which is what "explicit" buys.
The product being element-wise across whole embeddings is what makes it
vector-wise rather than bit-wise, and the anchor checks both
properties: the degree of each layer, and that swapping in a bit-wise
mixing breaks the field structure.

**Why it resembles a CNN and an RNN.** The outer product of
:math:`X^{k-1}` with :math:`X^0` forms a tensor that is compressed by
:math:`W^{k,h}` exactly as a filter compresses a feature map, and
:math:`X^0` is reused at every layer as an RNN reuses its input --
which is where the "compressed" in the name comes from.

**The combination.** xDeepFM sums a linear term, the CIN, and a plain
DNN, so bounded-degree explicit interactions and arbitrary implicit
ones are both available and neither has to do the other's job.

References
----------
Lian, J., Zhou, X., Zhang, F., Chen, Z., Xie, X. & Sun, G. (2018)
"xDeepFM: Combining Explicit and Implicit Feature Interactions for
Recommender Systems", *Proceedings of the 24th ACM SIGKDD
International Conference on Knowledge Discovery and Data Mining (KDD
'18)*, 1754-1763, doi:10.1145/3219819.3220023, arXiv:1803.05170. The
abstract and Sec. 1: plain DNNs generate feature interactions
IMPLICITLY and at the BIT-WISE level, unlike the traditional FM
framework which models them at the VECTOR-WISE level, and there is no
theoretical conclusion on the maximum degree a DNN represents; the
Compressed Interaction Network generating interactions explicitly and
vector-wise; CIN sharing functionality with CNNs and RNNs; and the
combination of CIN with a classical DNN into xDeepFM, learning
bounded-degree interactions explicitly and arbitrary ones implicitly.

Rendle, S. (2010) "Factorization Machines", *ICDM 2010*, 995-1000,
doi:10.1109/ICDM.2010.127. The vector-wise tradition; implemented in
:mod:`fmFM`.

Guo, H., Tang, R., Ye, Y., Li, Z. & He, X. (2017) "DeepFM: A
Factorization-Machine based Neural Network for CTR Prediction",
*IJCAI 2017*, 1725-1731, doi:10.24963/ijcai.2017/239. The predecessor
combining an FM with a DNN.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["hadamard", "cin_layer", "cin", "interaction_degree",
           "xdeepfm_score"]

_EPS = 1e-12


def hadamard(a, b):
    r"""The element-wise product of two field embeddings.

    Vector-wise: whole embeddings interact coordinate by coordinate,
    and no coordinate of one field is mixed with a *different*
    coordinate of another.
    """
    x = [float(v) for v in k.vec(a)]
    y = [float(v) for v in k.vec(b)]
    if len(x) != len(y):
        raise ValueError("xdeep: embeddings differ in length (%d, %d)"
                         % (len(x), len(y)))
    return [x[i] * y[i] for i in range(len(x))]


def cin_layer(X_prev, X0, W):
    r"""One CIN layer: outer product with :math:`X^0`, then compress.

    ``W[h][i][j]`` weights the interaction of previous feature map
    :math:`i` with input field :math:`j` into output map :math:`h`.
    """
    P = [[float(v) for v in r] for r in k.mat(X_prev)]
    Z = [[float(v) for v in r] for r in k.mat(X0)]
    mp, m = len(P), len(Z)
    d = len(Z[0])
    if len(P[0]) != d:
        raise ValueError("xdeep: the feature maps and input fields "
                         "differ in embedding size")
    out = []
    for h in range(len(W)):
        acc = [0.0] * d
        for i in range(mp):
            for j in range(m):
                w = float(W[h][i][j])
                if w == 0.0:
                    continue
                hp = hadamard(P[i], Z[j])
                for a in range(d):
                    acc[a] += w * hp[a]
        out.append(acc)
    return out


def cin(X0, Ws):
    r"""Stack CIN layers; layer :math:`k` holds degree :math:`k+1`.

    Each layer's sum-pooled output is a separate contribution, so
    every degree is represented explicitly rather than mixed.
    """
    Z = [[float(v) for v in r] for r in k.mat(X0)]
    cur = Z
    layers, pooled = [], []
    for W in Ws:
        cur = cin_layer(cur, Z, W)
        layers.append(cur)
        pooled.extend(sum(row) for row in cur)
    return RichResult(payload={
        "estimate": pooled, "pooled": pooled, "layers": layers,
        "degrees": [i + 2 for i in range(len(Ws))],
        "n_layers": len(Ws),
        "method": "Compressed Interaction Network; Lian et al. (2018)",
        "note": "layer k contains exactly degree-(k+1) interactions, "
                "so the maximum degree is BOUNDED and known",
    })


def interaction_degree(layer_index):
    r"""The interaction degree a CIN layer represents.

    A DNN has no such statement available -- which is the paper's
    point about implicit interactions.
    """
    i = int(layer_index)
    if i < 0:
        raise ValueError("xdeep: the layer index cannot be negative")
    return {"layer": i, "degree": i + 2,
            "note": "explicit and bounded, unlike a DNN's implicit "
                    "interactions of unknown maximum degree"}


def xdeepfm_score(x_linear, w_linear, X0, Ws, w_cin, dnn_output=0.0,
                  w_dnn=1.0, bias=0.0):
    r"""Linear + CIN + DNN, summed.

    Each part does its own job: the CIN supplies bounded-degree
    explicit interactions, the DNN arbitrary implicit ones.
    """
    xl = [float(v) for v in k.vec(x_linear)]
    wl = [float(v) for v in k.vec(w_linear)]
    if len(xl) != len(wl):
        raise ValueError("xdeep: the linear term is mis-sized")
    lin = sum(xl[i] * wl[i] for i in range(len(xl)))
    c = cin(X0, Ws)["pooled"]
    wc = [float(v) for v in k.vec(w_cin)]
    if len(wc) != len(c):
        raise ValueError("xdeep: %d CIN weights for %d pooled units"
                         % (len(wc), len(c)))
    ci = sum(wc[i] * c[i] for i in range(len(c)))
    z = float(bias) + lin + ci + float(w_dnn) * float(dnn_output)
    return {"logit": z,
            "probability": 1.0 / (1.0 + math.exp(-z))
            if z > -700 else 0.0,
            "linear": lin, "cin": ci,
            "dnn": float(w_dnn) * float(dnn_output),
            "note": "explicit and implicit interactions side by side, "
                    "neither doing the other's job"}


def cheatsheet():
    return ("xdeep: a DNN represents interactions IMPLICITLY and "
            "BIT-WISE -- mixing individual embedding coordinates "
            "across fields, with no statement about the maximum "
            "degree. FMs work VECTOR-WISE, whole embedding against "
            "whole embedding. CIN does that explicitly, one degree per "
            "layer: layer k = degree k+1, built by a Hadamard product "
            "with X^0 and then compressed -- CNN-like compression, "
            "RNN-like reuse of the input. xDeepFM sums linear + CIN + "
            "DNN so bounded explicit and arbitrary implicit "
            "interactions coexist.")


# compact alias per ledger/NAMING.md
xdeepfm = cin
