# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Autoformer decomposition and Auto-Correlation.

Wu, Xu, Wang and Long (2021) NeurIPS, arXiv:2106.13008, eqs. (1), (5), (6)
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["autoform", "joseph_autoformer"]

_METHOD = "Autoformer decomposition and Auto-Correlation"


def autoform(q, k, v, kernel=3, c=1.0):
    """Autoformer decomposition and Auto-Correlation.

    Autoformer decomposition plus Auto-Correlation, eqs. (1), (5), (6).

    Quoted from the paper:
        (5)  "R_XX(tau) = lim_{L->inf} (1/L) sum_{t=1..L} X_t X_{t-tau}"
        (6)  "tau_1,...,tau_k = arg Topk_{tau in {1..L}}(R_{Q,K}(tau))"
             with "k = floor(c x log L)"
             "Rhat_{Q,K}(tau_1),...,Rhat_{Q,K}(tau_k)
                  = SoftMax(R_{Q,K}(tau_1),...,R_{Q,K}(tau_k))"
             "Auto-Correlation(Q,K,V)
                  = sum_{i=1..k} Roll(V, tau_i) Rhat_{Q,K}(tau_i)"

    -- Wu, H., Xu, J., Wang, J. and Long, M., "Autoformer", NeurIPS
    2021 (arXiv:2106.13008).  ``Roll`` is the circular shift the paper
    uses to align sub-series; ties in the Topk are broken by the
    smaller lag so the selection is deterministic.

    Parameters
    ----------
    q : as documented for the shelf core
        See ``morie.fn._joseph.autoform``.
    k : as documented for the shelf core
        See ``morie.fn._joseph.autoform``.
    v : as documented for the shelf core
        See ``morie.fn._joseph.autoform``.
    kernel : as documented for the shelf core
        See ``morie.fn._joseph.autoform``.
    c : as documented for the shelf core
        See ``morie.fn._joseph.autoform``.

    Returns
    -------
    result : RichResult
        Payload keys: k, L, outmean, outmax, trendmean.

    References
    ----------
    Wu, Xu, Wang and Long (2021) NeurIPS, arXiv:2106.13008, eqs. (1), (5), (6)
    """
    res = _core.autoform(q=q, k=k, v=v, kernel=kernel, c=c)
    return RichResult(
        title=_METHOD,
        summary_lines=[("k", res["k"]), ("L", res["L"]), ("outmean", res["outmean"]), ("outmax", res["outmax"]), ("trendmean", res["trendmean"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_autoformer = autoform


def cheatsheet():
    return "autoform: Autoformer decomposition and Auto-Correlation"
