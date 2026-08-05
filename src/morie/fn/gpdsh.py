# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Density-shift detection on a stream by Kullback-Leibler divergence.

The stub carried the label "Gulenko et al (2019)".  No paper by that
author group in that year on density-shift detection could be matched
against Crossref (the closest, Gulenko et al 2018, CloudNet,
doi:10.1109/CloudNet.2018.8549546, is packet-level anomaly detection
for black-box services, a different problem).  The attribution is
therefore recorded as UNVERIFIED and the method below is implemented
from the formula the stub states, using the closed-form Gaussian
divergence of Kullback and Leibler (1951), Ann. Math. Statist.
22(1):79-86, doi:10.1214/aoms/1177729694:

    KL(p || q) = log(s_q/s_p) + (s_p^2 + (m_p - m_q)^2)/(2 s_q^2) - 1/2

between successive windows of the stream, each summarised by its
Gaussian marginal.  A shift is declared where that divergence exceeds
tau.  KL is not symmetric, and is zero exactly when the two windows
agree -- both are checked.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["gp_density_shift"]


def _kl(mp, sp, mq, sq):
    return math.log(sq / sp) + (sp * sp + (mp - mq) ** 2) / (2.0 * sq * sq) - 0.5


def gp_density_shift(y_stream, window=10, tau=0.5, floor=1e-12):
    """Per-step KL between consecutive windows, and the flagged shifts."""
    v = core.vec(y_stream)
    n = len(v)
    w = int(window)
    if w < 2:
        raise ValueError("gp_density_shift: window must be at least 2")
    if n < 2 * w:
        raise ValueError("gp_density_shift: the stream is shorter than two windows")
    t = float(tau)
    if t < 0:
        raise ValueError("gp_density_shift: tau must be non-negative")
    kls = []
    flags = []
    pos = []
    for s in range(w, n - w + 1):
        a = v[s - w:s]
        b = v[s:s + w]
        ma = sum(a) / w
        mb = sum(b) / w
        va = sum((x - ma) ** 2 for x in a) / (w - 1.0)
        vb = sum((x - mb) ** 2 for x in b) / (w - 1.0)
        sa = math.sqrt(max(va, float(floor)))
        sb = math.sqrt(max(vb, float(floor)))
        d = _kl(mb, sb, ma, sa)
        kls.append(d)
        flags.append(1 if d > t else 0)
        pos.append(s)
    mx = 0
    for i in range(len(kls)):
        if kls[i] > kls[mx]:
            mx = i
    return RichResult(
        title="Density shift by KL divergence",
        summary_lines=[("stream", n), ("window", w), ("shifts", sum(flags))],
        payload={
            "estimate": kls[mx],
            "kl": kls,
            "flagged": flags,
            "position": pos,
            "max_kl": kls[mx],
            "change_point": pos[mx],
            "n_shifts": sum(flags),
            "n": n,
            "method": "Gaussian KL between consecutive windows against tau; attribution 'Gulenko et al (2019)' UNVERIFIED",
        },
    )


def cheatsheet():
    return "gpdsh: density-shift detection by KL divergence"


# compact alias per ledger/NAMING.md
gpdensityshift = gp_density_shift
