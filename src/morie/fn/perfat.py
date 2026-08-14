# morie.fn -- function file (rootcoder007/morie)
r"""Performer: FAVOR+ kernel attention with positive random features.

Softmax attention costs :math:`O(L^2)` because it materialises
:math:`\exp(QK^\top)`. FAVOR+ never forms that matrix. It finds a
feature map :math:`\phi` whose inner product *is* the softmax kernel in
expectation,

.. math:: \mathrm{SM}(x, y) = \exp(x^\top y)
          = \mathbb{E}_{\omega\sim N(0, I_d)}
            \big[\phi_\omega(x)^\top \phi_\omega(y)\big],

and then reassociates :math:`(\phi(Q)\phi(K)^\top)V` into
:math:`\phi(Q)(\phi(K)^\top V)`, which is linear in :math:`L`.

**The features must be positive, and that is the paper's actual
contribution.** The obvious trigonometric map built from
:math:`\sin/\cos` is also unbiased, and it is unusable: attention
coefficients are a convex combination, so the kernel scores must be
non-negative, and a :math:`\sin/\cos` estimator has *large* variance
exactly where the true score approaches zero -- which is most entries.
It produces negative renormalisers and training collapses. Lemma 1
gives the fix,

.. math:: \phi(x) = \frac{\exp(-\|x\|^2/2)}{\sqrt{m}}
          \big(\exp(\omega_1^\top x), \dots,
          \exp(\omega_m^\top x)\big),

whose entries are positive by construction and whose variance goes to
zero as the estimated value does. Both maps are implemented, and the
anchor measures the variance of each near zero rather than repeating the
claim.

**Orthogonal features reduce variance further.** Drawing the
:math:`\omega` as exactly orthogonal rows (renormalised to chi
lengths) leaves the estimator unbiased and lowers its variance, which
is measurable and measured.

References
----------
Choromanski, K., Likhosherstov, V., Dohan, D., Song, X., Gane, A.,
Sarlos, T., Hawkins, P., Davis, J., Mohiuddin, A., Kaiser, L., Belanger,
D., Colwell, L. & Weller, A. (2021) "Rethinking Attention with
Performers", *International Conference on Learning Representations*,
arXiv:2009.14794. Eq. (6)-(7), Lemma 1, and the FAVOR+ mechanism.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez,
A. N., Kaiser, L. & Polosukhin, I. (2017) "Attention Is All You Need",
*Advances in Neural Information Processing Systems* 30,
arXiv:1706.03762. The quadratic attention this approximates.

Rahimi, A. & Recht, A. (2008) "Random Features for Large-Scale Kernel
Machines", *Advances in Neural Information Processing Systems* 20. The
random-feature idea, and the trigonometric map Lemma 1 replaces.

Yu, F. X., Suresh, A. T., Choromanski, K., Holtmann-Rice, D. & Kumar, S.
(2016) "Orthogonal Random Features", *Advances in Neural Information
Processing Systems* 29, arXiv:1610.09072. The orthogonality that lowers
the variance.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["favor_features", "favor_attention", "softmax_attention",
           "draw_projections", "kernel_estimate"]

_EPS = 1e-9


def _dot(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))


def _norm2(a):
    return sum(v * v for v in a)


def draw_projections(m, d, seed=0, orthogonal=True):
    r"""The :math:`\omega` rows.

    ``orthogonal=True`` Gram-Schmidts each block of :math:`d` rows and
    rescales them to chi-distributed lengths, which keeps the estimator
    unbiased -- each row is still marginally :math:`N(0, I_d)` -- while
    lowering its variance.
    """
    if m < 1 or d < 1:
        raise ValueError("perfat: need m >= 1 and d >= 1, got %d and %d"
                         % (m, d))
    rng = np.random.default_rng(seed)
    rows = [[rng.standard_normal() for _ in range(d)] for _ in range(m)]
    if not orthogonal:
        return rows
    out = []
    for start in range(0, m, d):
        block = rows[start:start + d]
        basis = []
        for v in block:
            u = list(v)
            for b in basis:
                p = _dot(u, b)
                u = [u[t] - p * b[t] for t in range(d)]
            nrm = math.sqrt(_norm2(u))
            if nrm < 1e-10:
                basis.append([1.0 if t == len(basis) else 0.0
                              for t in range(d)])
                continue
            basis.append([v2 / nrm for v2 in u])
        for t, b in enumerate(basis[:len(block)]):
            # restore the chi length of the original row, so each row is
            # still marginally N(0, I) and the estimator stays unbiased
            length = math.sqrt(_norm2(block[t]))
            out.append([v2 * length for v2 in b])
    return out[:m]


def favor_features(X, omegas, kind="positive", eps=1e-6):
    r"""Lemma 1's map, or the trigonometric one it replaces.

    ``"positive"`` is
    :math:`\phi(x) = m^{-1/2}\exp(-\|x\|^2/2)\exp(\omega^\top x)`;
    ``"trig"`` is the :math:`\sin/\cos` map, kept because its blow-up
    near zero is the reason Lemma 1 exists.
    """
    if kind not in ("positive", "trig"):
        raise ValueError("perfat: kind must be positive or trig, got %r"
                         % (kind,))
    m = len(omegas)
    out = []
    for x in X:
        nx = _norm2(x)
        proj = [_dot(w, x) for w in omegas]
        if kind == "positive":
            # the maximum is subtracted inside the exponential and put
            # back outside, so a large projection cannot overflow
            mx = max(proj) if proj else 0.0
            scale = math.exp(-0.5 * nx + mx) / math.sqrt(m)
            out.append([scale * math.exp(p - mx) + eps for p in proj])
        else:
            scale = math.exp(0.5 * nx) / math.sqrt(m)
            out.append([scale * math.sin(p) for p in proj]
                       + [scale * math.cos(p) for p in proj])
    return out


def kernel_estimate(x, y, omegas, kind="positive"):
    """The estimated softmax kernel between two vectors."""
    fx, fy = favor_features([x, y], omegas, kind=kind)
    return _dot(fx, fy)


def softmax_attention(Q, K, V, causal=False):
    """The quadratic attention FAVOR+ approximates, for comparison."""
    L = len(Q)
    out = []
    for i in range(L):
        lim = i + 1 if causal else L
        scores = [_dot(Q[i], K[j]) for j in range(lim)]
        mx = max(scores)
        w = [math.exp(s - mx) for s in scores]
        tot = sum(w)
        out.append([sum(w[j] * V[j][c] for j in range(lim)) / tot
                    for c in range(len(V[0]))])
    return out


def favor_attention(Q, K, V, n_features=128, seed=0, kind="positive",
                    orthogonal=True, causal=False):
    r"""Linear attention: :math:`\phi(Q)\,(\phi(K)^\top V)`.

    The reassociation is the whole point -- the L-by-L matrix is never
    formed, so the cost is :math:`O(Lmd)` rather than
    :math:`O(L^2 d)`.
    """
    Qm, Km, Vm = k.mat(Q), k.mat(K), k.mat(V)
    L, d = len(Qm), len(Qm[0]) if Qm else 0
    if len(Km) != len(Vm):
        raise ValueError("perfat: %d keys but %d values"
                         % (len(Km), len(Vm)))
    if len(Km) != L and not causal:
        raise ValueError("perfat: %d queries but %d keys" % (L, len(Km)))
    if d == 0 or len(Km[0]) != d:
        raise ValueError("perfat: query and key dimensions differ")
    om = draw_projections(int(n_features), d, seed=seed,
                          orthogonal=orthogonal)
    Qf = favor_features(Qm, om, kind=kind)
    Kf = favor_features(Km, om, kind=kind)
    dv = len(Vm[0])
    mf = len(Qf[0])
    out = []
    if not causal:
        # KV is m-by-dv and Ksum is m: both built once, then reused
        KV = [[sum(Kf[j][a] * Vm[j][c] for j in range(len(Kf)))
               for c in range(dv)] for a in range(mf)]
        Ksum = [sum(Kf[j][a] for j in range(len(Kf)))
                for a in range(mf)]
        for i in range(L):
            num = [sum(Qf[i][a] * KV[a][c] for a in range(mf))
                   for c in range(dv)]
            den = sum(Qf[i][a] * Ksum[a] for a in range(mf))
            if abs(den) < _EPS:
                raise ValueError("perfat: a renormaliser vanished at "
                                 "query %d; this is what the trig map "
                                 "does and Lemma 1 prevents" % i)
            out.append([v / den for v in num])
    else:
        KV = [[0.0] * dv for _ in range(mf)]
        Ksum = [0.0] * mf
        for i in range(L):
            for a in range(mf):
                Ksum[a] += Kf[i][a]
                for c in range(dv):
                    KV[a][c] += Kf[i][a] * Vm[i][c]
            num = [sum(Qf[i][a] * KV[a][c] for a in range(mf))
                   for c in range(dv)]
            den = sum(Qf[i][a] * Ksum[a] for a in range(mf))
            if abs(den) < _EPS:
                raise ValueError("perfat: a renormaliser vanished at "
                                 "query %d" % i)
            out.append([v / den for v in num])
    return RichResult(payload={
        "estimate": out, "output": out, "n_features": int(n_features),
        "kind": kind, "orthogonal": bool(orthogonal), "causal": causal,
        "L": L, "d": d, "d_v": dv,
        "method": "FAVOR+ linear attention, Choromanski et al. (2021) "
                  "Lemma 1",
    })


def cheatsheet():
    return ("perfat: phi(x) = exp(-|x|^2/2) exp(omega'x)/sqrt(m) makes "
            "E[phi(x)'phi(y)] = exp(x'y) EXACTLY (Lemma 1), so "
            "phi(Q)(phi(K)'V) replaces the L-by-L matrix. The features "
            "must be POSITIVE: the sin/cos map is also unbiased but its "
            "variance explodes where the kernel is near zero, which is "
            "most of it, and the renormaliser goes negative.")


# compact alias per ledger/NAMING.md
favorattention = favor_attention

# public names resolved by fn/_lazy_map.json
performer_favor_attention = favor_attention
