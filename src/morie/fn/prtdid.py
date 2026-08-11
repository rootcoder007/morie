# morie.fn -- wave3 slice w5_08 (rootcoder007/morie)
"""Goodman-Bacon 2x2 partition of TWFE -- alias of :mod:`morie.fn.gbacon`.

Goodman-Bacon, A. (2021), "Difference-in-differences with variation in
treatment timing", Journal of Econometrics 225(2):254-277,
doi:10.1016/j.jeconom.2021.03.014. Source used: NBER Working Paper
25018 (September 2018 version), local copy
/run/media/rootcoder/WD_BLACK/library/pdf/fetched-wave3/
goodman-bacon-2021-did-variation-treatment-timing.pdf. Implemented:
Theorem 1 (the DD decomposition theorem) -- the TWFE coefficient is
the weighted average over the complete PARTITION of the panel into
timing-group 2x2 designs, eqs. (7)-(9) for the three 2x2 estimators
(treated vs never; early vs late during the late group's pre-period;
late vs early using the already-treated early group) with the
variance-based weights s_kU, s_kl mu_kl, s_kl (1 - mu_kl) printed
just below them, summing to 1.

The "partition-based DiD" of the stub blurb IS this decomposition:
the partition is Theorem 1's exhaustive set of 2x2 cells. It is
already implemented in :mod:`morie.fn.gbacon` as
``goodman_bacon_decomp``; this module re-exports it rather than
carrying a second copy. ``prtdid`` and the stub-era name
``partition_did`` bind to the SAME callable (exact-zero alias).

R arms in both trees: ``morie_did_bacon_decomposition``
(r-package/morie/R/did.R and r-morie-oss mirror) plus the thin
documented alias ``Prtdid`` (R/prtdid_native.R, bit-identical in both
trees), anchored against the hand-computed fixture weights
(0.3, 0.4, 0.1, 0.2) and 2x2 estimates (2.25, 1.75, 1.25, 0.25) in
test-did-parity.R, with the Theorem 1 identity (weights sum to 1,
recomposition equals the TWFE coefficient 1.55) checked as numbers.
"""

from __future__ import annotations

from .gbacon import goodman_bacon_decomp  # noqa: F401

__all__ = ["prtdid", "partition_did", "goodman_bacon_decomp"]

# primary name = module name; stub-era long name kept as alias.
prtdid = goodman_bacon_decomp
partition_did = goodman_bacon_decomp


def cheatsheet():
    return ("prtdid: Goodman-Bacon (2021) Theorem 1 partition of the "
            "TWFE DiD into weighted 2x2s -- alias of "
            "gbacon.goodman_bacon_decomp")
