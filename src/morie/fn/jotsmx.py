# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""TSMixer time-mixing and feature-mixing.

Chen, Li, Yoder, Arik and Pfister (2023) TMLR, arXiv:2303.06053, eqs. (4)-(5)
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["tsmixer", "joseph_tsmixer"]

_METHOD = "TSMixer time-mixing and feature-mixing"


def tsmixer(x, wtime, btime, wfeat, bfeat, wproj, bproj, horizon):
    """TSMixer time-mixing and feature-mixing.

    TSMixer time-mixing and feature-mixing, all-MLP.

    Quoted from the paper:
        (4)  "TP_{L->T}(X)_{*,i} = W_1 X_{*,i} + b_1, for all i = 1..C"
        (5)  "TM(X)_{*,i} = Norm(X_{*,i} + Drop(sigma(TP_{L->L}(X)_{*,i})))"

    -- Chen, S.-A., Li, C.-L., Yoder, N. C., Arik, S. O. and Pfister,
    T., "TSMixer: An All-MLP Architecture for Time Series Forecasting",
    TMLR 2023 (arXiv:2303.06053), Appendix B.3.1.  The paper describes
    feature mixing and the 2D normalization in prose rather than as
    numbered equations; both are implemented from that prose and said
    to be so here.  Dropout is omitted because it is a training-time
    stochastic operation and this routine is evaluation-time and
    deterministic.

    ``x`` is a list of C channels each of length L.

    Parameters
    ----------
    x : as documented for the shelf core
        See ``morie.fn._joseph.tsmixer``.
    wtime : as documented for the shelf core
        See ``morie.fn._joseph.tsmixer``.
    btime : as documented for the shelf core
        See ``morie.fn._joseph.tsmixer``.
    wfeat : as documented for the shelf core
        See ``morie.fn._joseph.tsmixer``.
    bfeat : as documented for the shelf core
        See ``morie.fn._joseph.tsmixer``.
    wproj : as documented for the shelf core
        See ``morie.fn._joseph.tsmixer``.
    bproj : as documented for the shelf core
        See ``morie.fn._joseph.tsmixer``.
    horizon : as documented for the shelf core
        See ``morie.fn._joseph.tsmixer``.

    Returns
    -------
    result : RichResult
        Payload keys: nchannels, L, horizon, mean, sumsq.

    References
    ----------
    Chen, Li, Yoder, Arik and Pfister (2023) TMLR, arXiv:2303.06053, eqs. (4)-(5)
    """
    res = _core.tsmixer(x=x, wtime=wtime, btime=btime, wfeat=wfeat, bfeat=bfeat, wproj=wproj, bproj=bproj, horizon=horizon)
    return RichResult(
        title=_METHOD,
        summary_lines=[("nchannels", res["nchannels"]), ("L", res["L"]), ("horizon", res["horizon"]), ("mean", res["mean"]), ("sumsq", res["sumsq"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_tsmixer = tsmixer


def cheatsheet():
    return "tsmixer: TSMixer time-mixing and feature-mixing"
