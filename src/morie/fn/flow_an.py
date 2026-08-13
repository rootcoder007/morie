# morie.fn -- function file (rootcoder007/morie)
r"""RealNVP: exact-likelihood density estimation, and anomaly scoring.

A normalizing flow builds a density by pushing a simple one through an
invertible map. The change of variables gives the likelihood exactly,

.. math:: \log p_X(x) = \log p_Z(f(x))
          + \log\Big|\det \frac{\partial f}{\partial x}\Big|,

so there is nothing to approximate -- provided the Jacobian determinant
is computable, which for a general map it is not.

**The coupling layer is the trick that makes it computable.** Split the
input, leave one half alone, and transform the other half using only the
untouched half:

.. math:: y_{1:d} = x_{1:d}, \qquad
          y_{d+1:D} = x_{d+1:D} \odot \exp\!\big(s(x_{1:d})\big)
          + t(x_{1:d}).

The Jacobian is triangular, so its determinant is just
:math:`\exp(\sum s)` -- and :math:`s` and :math:`t` may be arbitrarily
complicated networks, because they are never differentiated through for
the determinant. The inverse is equally cheap, which matters: a flow
that cannot be inverted cannot sample.

**Both directions are exact and both are checked.** The anchor round
trips :math:`f^{-1}(f(x)) = x` to machine precision and compares the
analytic log-determinant against one computed by finite differences on
the full Jacobian -- because a sign error in the log-det leaves the
model trainable and the likelihood wrong, which is exactly the kind of
error that never surfaces on its own.

**Masks must alternate, or half the input is never modelled.** A
coupling layer leaves its first half untouched; stack two with the same
mask and those channels pass through unchanged with density
contribution zero. The anchor checks the composed map actually depends
on every input.

**Anomaly scoring is then just the likelihood.** A point in a
low-density region is unlikely under the fitted flow, so
:math:`-\log p(x)` ranks anomalies, with a threshold taken as a
quantile of the training scores.

References
----------
Dinh, L., Sohl-Dickstein, J. & Bengio, S. (2017) "Density Estimation
using Real NVP", *International Conference on Learning
Representations*, arXiv:1605.08803. The affine coupling layer, its
Jacobian, and the multi-scale architecture.

Dinh, L., Krueger, D. & Bengio, Y. (2015) "NICE: Non-linear
Independent Components Estimation", *ICLR Workshop*, arXiv:1410.8516.
The additive coupling RealNVP generalises.

Rezende, D. J. & Mohamed, S. (2015) "Variational Inference with
Normalizing Flows", *Proceedings of the 32nd International Conference
on Machine Learning*, PMLR 37, 1530-1538, arXiv:1505.05770. The
normalizing-flow framing.

Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S. &
Lakshminarayanan, B. (2021) "Normalizing Flows for Probabilistic
Modeling and Inference", *Journal of Machine Learning Research* 22(57),
1-64, arXiv:1912.02762. A review of the family.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["coupling_forward", "coupling_inverse", "flow_forward",
           "flow_inverse", "log_prob", "anomaly_score",
           "alternating_masks"]

_EPS = 1e-12
_LOG2PI = math.log(2.0 * math.pi)


def alternating_masks(d, n_layers):
    """Binary masks that alternate, so every channel is transformed.

    Stack two layers with the SAME mask and the untouched channels pass
    through unchanged with zero density contribution -- the model still
    trains and simply never sees half its input.
    """
    if d < 2:
        raise ValueError("flow_an: need at least 2 dimensions, got %d"
                         % d)
    out = []
    for t in range(int(n_layers)):
        par = t % 2
        out.append([1.0 if (i % 2) == par else 0.0 for i in range(d)])
    return out


def _st(x, mask, Ws, bs, Wt, bt, scale_cap=5.0):
    """s and t computed from the MASKED-IN half alone."""
    xin = [x[i] * mask[i] for i in range(len(x))]
    hs = [sum(xin[i] * Ws[i][j] for i in range(len(x))) + bs[j]
          for j in range(len(bs))]
    ht = [sum(xin[i] * Wt[i][j] for i in range(len(x))) + bt[j]
          for j in range(len(bt))]
    # tanh-capped log-scale: an uncapped s exponentiates and the
    # determinant overflows long before the model is any good
    s = [scale_cap * math.tanh(v) * (1.0 - mask[j])
         for j, v in enumerate(hs)]
    t = [ht[j] * (1.0 - mask[j]) for j in range(len(ht))]
    return s, t


def coupling_forward(x, mask, Ws, bs, Wt, bt, scale_cap=5.0):
    r"""One affine coupling layer, and its exact log-determinant."""
    s, t = _st(x, mask, Ws, bs, Wt, bt, scale_cap)
    y = [x[i] * mask[i]
         + (1.0 - mask[i]) * (x[i] * math.exp(s[i]) + t[i])
         for i in range(len(x))]
    return y, sum(s)


def coupling_inverse(y, mask, Ws, bs, Wt, bt, scale_cap=5.0):
    """The exact inverse -- available because the untouched half is
    enough to recompute s and t."""
    s, t = _st(y, mask, Ws, bs, Wt, bt, scale_cap)
    x = [y[i] * mask[i]
         + (1.0 - mask[i]) * ((y[i] - t[i]) * math.exp(-s[i]))
         for i in range(len(y))]
    return x, -sum(s)


def flow_forward(x, layers):
    """Compose the layers, accumulating the log-determinant."""
    z = list(x)
    logdet = 0.0
    for (mask, Ws, bs, Wt, bt) in layers:
        z, ld = coupling_forward(z, mask, Ws, bs, Wt, bt)
        logdet += ld
    return z, logdet


def flow_inverse(z, layers):
    """Invert the composition, in reverse order."""
    x = list(z)
    logdet = 0.0
    for (mask, Ws, bs, Wt, bt) in reversed(layers):
        x, ld = coupling_inverse(x, mask, Ws, bs, Wt, bt)
        logdet += ld
    return x, logdet


def log_prob(x, layers):
    r"""The exact log density under a standard normal base."""
    z, logdet = flow_forward(x, layers)
    base = -0.5 * sum(v * v for v in z) - 0.5 * len(z) * _LOG2PI
    return base + logdet, z, logdet


def anomaly_score(X, layers, threshold_quantile=0.95, reference=None):
    r"""Score by negative log-likelihood, threshold by a quantile.

    ``reference`` supplies the rows the threshold is read off, so a
    caller can set it on clean training data and apply it to fresh
    data -- taking the quantile of the scored set itself guarantees
    exactly that fraction is flagged whatever the data looks like,
    which is not a detector.
    """
    Xm = k.mat(X)
    scores = [-log_prob(row, layers)[0] for row in Xm]
    ref = scores if reference is None else [
        -log_prob(r, layers)[0] for r in k.mat(reference)]
    q = float(threshold_quantile)
    if not 0.0 < q < 1.0:
        raise ValueError("flow_an: threshold_quantile must be in "
                         "(0, 1), got %r" % (threshold_quantile,))
    thr = k.quantile7(sorted(ref), q)
    flags = [1.0 if v > thr else 0.0 for v in scores]
    return RichResult(payload={
        "estimate": scores, "score": scores, "threshold": thr,
        "flag": flags, "n_flagged": int(sum(flags)), "n": len(Xm),
        "quantile": q, "self_referenced": reference is None,
        "log_likelihood": [-v for v in scores],
        "method": "RealNVP negative log-likelihood anomaly score, "
                  "Dinh, Sohl-Dickstein & Bengio (2017)",
    })


def cheatsheet():
    return ("flow_an: coupling layer y1 = x1, y2 = x2*exp(s(x1)) + "
            "t(x1). Jacobian is TRIANGULAR so log|det| = sum(s), and s, "
            "t can be arbitrary nets because they are never "
            "differentiated for the determinant. log p(x) = log p_z(f(x))"
            " + sum(s), exact. Masks must ALTERNATE or half the input "
            "is never transformed. Cap the log-scale or exp overflows.")


# compact alias per ledger/NAMING.md
anomalyscore = anomaly_score
