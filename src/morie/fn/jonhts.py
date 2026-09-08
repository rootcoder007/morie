# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""N-HiTS multi-rate sampling and hierarchical interpolation.

Challu et al. (2023) AAAI, arXiv:2201.12886, eqs. (1)-(4)
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["nhitsnet", "joseph_nhits"]

_METHOD = "N-HiTS multi-rate sampling and hierarchical interpolation"


def nhitsnet(y, horizon, kernels, ratios, wf, wb):
    """N-HiTS multi-rate sampling and hierarchical interpolation.

    N-HiTS multi-rate sampling with hierarchical interpolation.

    Quoted from the paper:
        (1)  "y^(p)_{t-L:t,l} = MaxPool(y_{t-L:t,l}, k_l)"
        (2)  "h_l = MLP_l(y^(p)_{t-L:t,l}); theta^f_l = LINEAR^f(h_l);
              theta^b_l = LINEAR^b(h_l)"
        (3)  "yhat_{tau,l} = g(tau, theta^f_l) ... ytilde_{tau,l}
              = g(tau, theta^b_l)"   with "|theta^f_l| = ceil(r_l H)"
        (4)  "g(tau, theta) = theta[t1]
              + ((theta[t2] - theta[t1])/(t2 - t1))(tau - t1)"
        doubly residual stacking:
             "yhat_{t+1:t+H} = sum_l yhat_{t+1:t+H,l};
              y_{t-L:t,l+1} = y_{t-L:t,l} - ytilde_{t-L:t,l}"

    -- Challu, C., Olivares, K. G., Oreshkin, B. N., Garza, F.,
    Mergenthaler-Canseco, M. and Dubrawski, A., "N-HiTS: Neural
    Hierarchical Interpolation for Time Series Forecasting", AAAI 2023
    (arXiv:2201.12886).

    ``wf[l]`` and ``wb[l]`` stand in for the paper's MLP_l followed by
    LINEAR^f / LINEAR^b: a single caller-supplied linear map from the
    pooled window to the coefficients.  That collapse is stated here
    rather than hidden -- it is what makes the block deterministic
    without a trained network -- and the expressivity ratio ``r_l``
    still governs how many coefficients each block gets, which is the
    hierarchical part the paper is actually about.

    Parameters
    ----------
    y : as documented for the shelf core
        See ``morie.fn._joseph.nhitsnet``.
    horizon : as documented for the shelf core
        See ``morie.fn._joseph.nhitsnet``.
    kernels : as documented for the shelf core
        See ``morie.fn._joseph.nhitsnet``.
    ratios : as documented for the shelf core
        See ``morie.fn._joseph.nhitsnet``.
    wf : as documented for the shelf core
        See ``morie.fn._joseph.nhitsnet``.
    wb : as documented for the shelf core
        See ``morie.fn._joseph.nhitsnet``.

    Returns
    -------
    result : RichResult
        Payload keys: nblocks, first, last, mean, residnorm.

    References
    ----------
    Challu et al. (2023) AAAI, arXiv:2201.12886, eqs. (1)-(4)
    """
    res = _core.nhitsnet(y=y, horizon=horizon, kernels=kernels, ratios=ratios, wf=wf, wb=wb)
    return RichResult(
        title=_METHOD,
        summary_lines=[("nblocks", res["nblocks"]), ("first", res["first"]), ("last", res["last"]), ("mean", res["mean"]), ("residnorm", res["residnorm"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_nhits = nhitsnet


def cheatsheet():
    return "nhitsnet: N-HiTS multi-rate sampling and hierarchical interpolation"
