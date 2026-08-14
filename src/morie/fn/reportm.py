# morie.fn -- function file (rootcoder007/morie)
"""Report Noisy Max -- Dwork & Roth (2014), section 3.3."""

from __future__ import annotations

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["reportm", "report_noisy_max"]


def report_noisy_max(counts, epsilon, sensitivity=1.0, seed=1):
    r"""Privately select the index of the largest counting query.

    Adds independent Laplace noise with scale
    :math:`\Delta/\varepsilon` to each count and returns the index of
    the largest noisy count.  For monotone sensitivity-1 counting
    queries this is Dwork-Roth Claim 3.9: the released *index* is
    (epsilon, 0)-differentially private, even though the full noisy
    count vector would not be (its L1 sensitivity is m).  The winning
    noisy count may be released at no extra privacy cost (text after
    Claim 3.9); the losing noisy counts must not be.

    Determinism: noise comes from the shared Lehmer minstd stream, so a
    given seed reproduces the same selection in both language arms.
    Laplace draws by inverse CDF,
    ``x = -b sign(u - 1/2) ln(1 - 2|u - 1/2|)``.
    Ties: the first maximum in scan order wins in both arms.

    Parameters
    ----------
    counts : array-like
        Values of the m counting queries.
    epsilon : float
        Privacy budget, positive.
    sensitivity : float, default 1.0
        Sensitivity of each individual query (1 for counting queries;
        the Claim 3.9 guarantee is stated for monotone sensitivity-1
        queries).
    seed : int, default 1
        Seed for the shared deterministic stream.

    Returns
    -------
    RichResult
        ``index`` (0-based argmax), ``winner`` (winning noisy count),
        ``epsilon``, ``scale``, ``n``.

    References
    ----------
    Dwork, C., & Roth, A. (2014). The algorithmic foundations of
        differential privacy. *FnT-TCS*, 9(3-4), 211-487. Section 3.3,
        Report Noisy Max and Claim 3.9 (noise Lap(1/epsilon), release
        the argmax index only).
        Local source: /run/media/rootcoder/WD_BLACK/library/pdf/fetched-wave3/dwork-roth-2014-algorithmic-foundations-differential-privacy.pdf
    """
    x = [float(v) for v in counts]
    n = len(x)
    if n == 0:
        raise ValueError("counts must be non-empty")
    eps = float(epsilon)
    if eps <= 0.0:
        raise ValueError("epsilon must be positive")
    b = float(sensitivity) / eps
    if b <= 0.0:
        raise ValueError("sensitivity must be positive")
    g = C.Lcg(seed)
    best = -math.inf
    idx = -1
    for i in range(n):
        u = g.unif()
        h = u - 0.5
        s = 1.0 if h > 0 else (-1.0 if h < 0 else 0.0)
        noisy = x[i] - b * s * math.log(1.0 - 2.0 * abs(h))
        if noisy > best:
            best = noisy
            idx = i
    return RichResult(payload={
        "index": float(idx), "winner": best, "estimate": float(idx),
        "epsilon": eps, "scale": b, "n": n,
        "method": "Report Noisy Max (Dwork-Roth 2014, Claim 3.9)"})


#: Primary name for the module.
reportm = report_noisy_max


def cheatsheet():
    return "reportm: Report Noisy Max index selection (Dwork-Roth 2014, Claim 3.9)."

# public names resolved by fn/_lazy_map.json
reportnoisymax = report_noisy_max
