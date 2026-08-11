# morie.fn -- wave3 slice w5_08 (rootcoder007/morie)
"""Callaway-Sant'Anna group-time ATT -- alias of :mod:`morie.fn.cssant`.

Callaway, B. and Sant'Anna, P. H. C. (2021), "Difference-in-Differences
with multiple time periods", Journal of Econometrics 225(2):200-230,
doi:10.1016/j.jeconom.2020.12.001. Source used: arXiv:1803.09015v4,
local copy /run/media/rootcoder/WD_BLACK/library/pdf/fetched-wave3/
callaway-santanna-2021-did-multiple-time-periods.pdf. The unconditional
estimands implemented are eq. (2.8) (never-treated comparison) and
eq. (2.9) (not-yet-treated comparison, evaluated at max(t, g) as in the
authors' `did` R package), with base period g-1 (Theorem 1, delta = 0);
aggregations follow Section 3.

The estimator is already implemented in :mod:`morie.fn.cssant` under
``callaway_santanna`` / ``group_time_att`` / ``aggregate_att``; this
module re-exports it rather than carrying a second copy (wave2
DUPMAP discipline: aliases beat re-implementations). ``causdidcs``
and the stub-era name ``causal_did_callaway_sa`` are bindings to the
SAME callable, so they agree with the target at exactly zero.

The R arms live in both trees as ``morie_did_group_time_att`` +
``morie_did_aggregate_gt_att`` (r-package/morie/R/did.R and
r-morie-oss/R/did.R), anchored cell-by-cell against
``morie.fn.cssant`` on the hand-designed staggered fixture in
tests/testthat/test-did-parity.R (true overall ATT 2.05 by design).
NOTE: the Python ``se`` uses the influence-function aggregation of
Callaway-Sant'Anna Section 3; the R machinery's aggregate ``std_error``
uses the independent-cells approximation sqrt(sum(w_i^2 se_i^2)).
Point estimates, ATT(g,t) cells and event-study coefficients are the
parity surface; the two se conventions predate this alias and are
documented, not reconciled, here.
"""

from __future__ import annotations

from .cssant import (  # noqa: F401
    aggregate_att,
    callaway_santanna,
    group_time_att,
)

__all__ = ["causdidcs", "causal_did_callaway_sa", "callaway_santanna",
           "group_time_att", "aggregate_att"]

# primary name = module name; stub-era long name kept as alias.
causdidcs = callaway_santanna
causal_did_callaway_sa = callaway_santanna


def cheatsheet():
    return ("causdidcs: Callaway-Sant'Anna (2021) ATT(g,t) -- alias of "
            "cssant.callaway_santanna (eq. 2.8/2.9, base period g-1)")
