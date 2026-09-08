# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Gaussian mechanism for differential privacy.

Dwork and Roth (2014), *The Algorithmic Foundations of Differential
Privacy*, Foundations and Trends in Theoretical Computer Science
9(3-4):211-407, doi:10.1561/0400000042, Theorem A.1 (appendix A, cited
in section 3.5): for eps in (0, 1) and c^2 > 2 ln(1.25 / delta), the
mechanism

    M(D) = f(D) + N(0, sigma^2),   sigma >= c Delta_2 f / eps

is (eps, delta)-differentially private, where Delta_2 f is the L2
sensitivity of f.  The bound is only valid for eps < 1, which is
checked rather than assumed; the noise itself is drawn from the
deterministic inverse-normal stream so the released value is
reproducible across language arms.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["gaussian_mechanism"]


def gaussian_mechanism(f_value, l2_sens, epsilon, delta, draw=1):
    """Noisy release, its scale, and the privacy parameters it satisfies."""
    fv = core.vec(f_value)
    if len(fv) == 0:
        raise ValueError("gaussian_mechanism: f_value is empty")
    s = float(l2_sens)
    e = float(epsilon)
    d = float(delta)
    if s <= 0:
        raise ValueError("gaussian_mechanism: the L2 sensitivity must be positive")
    if not 0 < e < 1:
        raise ValueError("gaussian_mechanism: Theorem A.1 requires 0 < epsilon < 1")
    if not 0 < d < 1:
        raise ValueError("gaussian_mechanism: delta must lie in (0, 1)")
    c = math.sqrt(2.0 * math.log(1.25 / d))
    sigma = c * s / e
    k = int(draw)
    noise = [sigma * core.qnorm(core.vdc(k + i, 2)) for i in range(len(fv))]
    out = [fv[i] + noise[i] for i in range(len(fv))]
    return RichResult(
        title="Gaussian mechanism",
        summary_lines=[("epsilon", e), ("delta", d), ("sigma", sigma)],
        payload={
            "estimate": out[0],
            "released": out,
            "noise": noise,
            "sigma": sigma,
            "c": c,
            "epsilon": e,
            "delta": d,
            "n": len(fv),
            "method": "f(D) + N(0, sigma^2) with sigma = sqrt(2 ln(1.25/delta)) Delta_2 f / eps, Dwork & Roth (2014) Thm A.1",
        },
    )


def cheatsheet():
    return "gaussm: Gaussian mechanism for differential privacy"
