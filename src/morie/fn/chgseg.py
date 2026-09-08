# SPDX-License-Identifier: AGPL-3.0-or-later
"""Penalised changepoint segmentation (PELT, Normal mean cost)."""

from . import _array_core as np  # noqa: F401  (kept for API uniformity)

from ._richresult import RichResult
from .pelt import pelt as _pelt

__all__ = ["chgseg", "changepoint_segmentation"]


def chgseg(y, penalty=None):
    """
    Changepoint segmentation of a univariate series by PELT with the
    Normal change-in-mean cost.

    This is the eq (3) optimal-partitioning objective of Killick,
    Fearnhead & Eckley (2012) solved exactly by their Algorithm 2
    (PELT), specialised to the twice-negative-log-likelihood cost for
    a change in mean with unit variance (sum of squared deviations
    from the segment mean). The computational core is shared with
    morie.fn.pelt.pelt (same paper, same algorithm).

    Parameters
    ----------
    y : array-like
        Series.
    penalty : float, optional
        beta; default log(n) (SIC with p = 1).

    Returns
    -------
    result : RichResult
        Keys as in pelt: changepoints, n_changepoints, objective,
        penalty, segment_means.

    References
    ----------
    Killick, R., Fearnhead, P. and Eckley, I. A. (2012), "Optimal
    detection of changepoints with a linear computational cost", JASA
    107(500), 1590-1598 (arXiv:1101.1438), eq (3), Algorithm 2.
    Source PDF: /run/media/rootcoder/WD_BLACK/library/pdf/fetched-wave3/
    killick-fearnhead-eckley-2012-pelt-optimal-changepoint-linear-cost.pdf
    """
    res = _pelt(y, cost="mean", penalty=penalty)
    out = dict(res)
    out["method"] = "PELT mean-change segmentation (Killick et al. 2012, eq 3)"
    return RichResult(payload=out)


def changepoint_segmentation(y, penalty=None):
    """Alias for chgseg (original stub export name)."""
    return chgseg(y, penalty=penalty)


def cheatsheet():
    return "chgseg(y, penalty) -> exact penalised mean-change segmentation via PELT"
