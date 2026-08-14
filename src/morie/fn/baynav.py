# morie.fn -- function file (rootcoder007/morie)
r"""Normalizing flows and automatic differentiation VI.

Variational inference replaces integration with optimisation, and its
accuracy is bounded by the family it optimises over. Two papers attack
that from opposite ends.

**Normalizing flows: make the family rich.** A simple :math:`q_0` is
pushed through invertible maps :math:`f_k`; the density transforms by
the change of variables,

.. math:: \log q_K(z_K) = \log q_0(z_0)
          - \sum_{k=1}^{K}\log\Big|\det\frac{\partial f_k}
          {\partial z_{k-1}}\Big|.

The determinant is the whole design problem. A general
:math:`d\times d` Jacobian costs :math:`O(d^3)`; a **planar** flow
:math:`f(z) = z + u\,h(w^\top z + b)` has a rank-one Jacobian, so by
the matrix determinant lemma its determinant is
:math:`1 + u^\top\psi(z)` -- computable in :math:`O(d)`. Cheap
determinants buy depth, and depth buys expressiveness.

**Invertibility is a constraint, not a hope.** A planar flow is
invertible only when :math:`u^\top w \ge -1`; outside that region the
change-of-variables formula is simply wrong, and the reported bound is
meaningless. ``enforce_invertibility`` projects :math:`u` back, and
the anchor confirms the constraint binds.

**ADVI: make the family automatic.** Rather than enriching the family,
Kucukelbir et al. remove the derivation. The model's constrained
parameters are mapped to :math:`\mathbb{R}^K` by a bijection
:math:`T`, a Gaussian is fitted *there*, and the Jacobian of
:math:`T^{-1}` corrects the density. The user writes only the model:
no gradients, no coordinate updates, no per-model derivation.

**Both optimise the same bound**, so ``elbo`` is shared -- and the
anchor uses it as the common yardstick: a flow of depth 0 must equal
plain mean-field, and adding layers must not lower the bound.

References
----------
Rezende, D. J. & Mohamed, S. (2015) "Variational Inference with
Normalizing Flows", *Proceedings of the 32nd International Conference
on Machine Learning (ICML 2015)*, PMLR 37, 1530-1538,
arXiv:1505.05770. The transformation of a simple density through a
sequence of invertible maps with the log-determinant correction;
planar and radial flows whose Jacobian determinants are computable in
linear time by the matrix determinant lemma; and the invertibility
condition on the planar flow parameters.

Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A. & Blei, D. M.
(2017) "Automatic Differentiation Variational Inference", *Journal of
Machine Learning Research* 18(14), 1-45, arXiv:1603.00788.
Transforming constrained latent variables to the real coordinate space
by a bijection, fitting a Gaussian there, and correcting with the
Jacobian; requiring only the model from the user.

Blei, D. M., Kucukelbir, A. & McAuliffe, J. D. (2017) "Variational
Inference: A Review for Statisticians", *Journal of the American
Statistical Association* 112(518), 859-877,
doi:10.1080/01621459.2017.1285773.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["planar_flow", "flow_log_density", "enforce_invertibility",
           "transform_to_real", "elbo"]

_EPS = 1e-12


def enforce_invertibility(u, w):
    r"""Project :math:`u` so that :math:`u^\top w \ge -1`.

    Not cosmetic: outside this region the flow is not invertible and
    the change-of-variables formula does not hold, so the bound being
    reported is not a bound.
    """
    uv = [float(v) for v in k.vec(u)]
    wv = [float(v) for v in k.vec(w)]
    if len(uv) != len(wv):
        raise ValueError("baynav: u and w differ in length")
    uw = sum(uv[i] * wv[i] for i in range(len(uv)))
    if uw >= -1.0:
        return {"u": uv, "adjusted": False, "u_dot_w": uw}
    m = -1.0 + math.log1p(math.exp(uw))
    nw = sum(v * v for v in wv)
    if nw <= _EPS:
        raise ValueError("baynav: w is zero, so the flow is "
                         "degenerate")
    out = [uv[i] + (m - uw) * wv[i] / nw for i in range(len(uv))]
    return {"u": out, "adjusted": True, "u_dot_w": uw,
            "u_dot_w_after": sum(out[i] * wv[i]
                                 for i in range(len(out))),
            "note": "u'w >= -1 is required for invertibility"}


def planar_flow(z, u, w, b):
    r""":math:`f(z) = z + u\,\tanh(w^\top z + b)`, with its
    log-determinant.

    The Jacobian is rank-one, so the matrix determinant lemma gives
    :math:`|1 + u^\top\psi(z)|` in :math:`O(d)` rather than
    :math:`O(d^3)`.
    """
    zv = [float(v) for v in k.vec(z)]
    fixed = enforce_invertibility(u, w)
    uv = fixed["u"]
    wv = [float(v) for v in k.vec(w)]
    a = sum(wv[i] * zv[i] for i in range(len(zv))) + float(b)
    t = math.tanh(a)
    out = [zv[i] + uv[i] * t for i in range(len(zv))]
    dt = 1.0 - t * t
    psi = [dt * wv[i] for i in range(len(wv))]
    det = 1.0 + sum(uv[i] * psi[i] for i in range(len(uv)))
    return {"z": out, "log_det": math.log(max(abs(det), _EPS)),
            "det": det, "invertibility_adjusted": fixed["adjusted"],
            "note": "rank-one Jacobian, so the determinant is O(d)"}


def flow_log_density(z0, log_q0, layers):
    r"""Push a sample through the flow and correct the density.

    :math:`\log q_K = \log q_0 - \sum_k \log|\det J_k|` -- the
    subtraction is the whole content of the change of variables.
    """
    z = [float(v) for v in k.vec(z0)]
    lq = float(log_q0)
    dets = []
    for (u, w, b) in layers:
        r = planar_flow(z, u, w, b)
        z = r["z"]
        dets.append(r["log_det"])
        lq -= r["log_det"]
    return RichResult(payload={
        "estimate": lq, "log_q": lq, "z": z,
        "log_dets": dets, "depth": len(layers),
        "method": "normalizing flow; Rezende & Mohamed (2015)",
        "note": "depth 0 leaves the density untouched, which is the "
                "mean-field case",
    })


def transform_to_real(value, support="positive", eps=1e-10):
    r"""ADVI's bijection to :math:`\mathbb{R}` with its log-Jacobian.

    Fitting a Gaussian to a positive parameter directly puts mass on
    impossible values; transforming first is what makes one automatic
    recipe cover every model.
    """
    v = float(value)
    if support == "positive":
        if v <= 0.0:
            raise ValueError("baynav: a positive parameter must be "
                             "positive, got %r" % (value,))
        return {"real": math.log(v), "log_jacobian": -math.log(v),
                "inverse": math.exp(math.log(v))}
    if support == "unit":
        if not 0.0 < v < 1.0:
            raise ValueError("baynav: a unit parameter must lie in "
                             "(0,1), got %r" % (value,))
        z = math.log(v / (1.0 - v))
        return {"real": z,
                "log_jacobian": -math.log(v) - math.log(1.0 - v),
                "inverse": 1.0 / (1.0 + math.exp(-z))}
    if support == "real":
        return {"real": v, "log_jacobian": 0.0, "inverse": v}
    raise ValueError("baynav: support must be positive, unit or real, "
                     "got %r" % (support,))


def elbo(log_joint, log_q, samples):
    r""":math:`E_q[\log p(x,z) - \log q(z)]`.

    The common yardstick: both approaches optimise this, so a deeper
    flow must not lower it.
    """
    if not samples:
        raise ValueError("baynav: no samples given")
    vals = [float(log_joint(s)) - float(log_q(s)) for s in samples]
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / max(len(vals) - 1, 1)
    return {"elbo": m, "se": math.sqrt(var / len(vals)),
            "n_samples": len(vals),
            "note": "a lower bound on the log evidence; enriching the "
                    "family can only raise it"}


def cheatsheet():
    return ("baynav: variational accuracy is capped by the FAMILY. Two "
            "fixes. NORMALIZING FLOWS enrich it: push q0 through "
            "invertible maps and subtract the log-determinants. The "
            "determinant is the design problem -- a planar flow's "
            "Jacobian is RANK ONE, so the lemma gives it in O(d) "
            "instead of O(d^3), and cheap determinants buy depth. "
            "Invertibility needs u'w >= -1, or the formula is simply "
            "wrong. ADVI instead makes it AUTOMATIC: map constrained "
            "parameters to R^K, fit a Gaussian there, correct by the "
            "Jacobian; the user writes only the model.")


# compact alias per ledger/NAMING.md
normalizingflow = flow_log_density
