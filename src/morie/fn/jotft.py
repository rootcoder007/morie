# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Temporal Fusion Transformer gating and variable selection.

Lim, Arik, Loeff and Pfister (2021) IJF 37(4), arXiv:1912.09363, eqs. (2)-(6), (23), (25)
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["tftnet", "joseph_temporal_fusion_transformer"]

_METHOD = "Temporal Fusion Transformer gating and variable selection"


def tftnet(a, w1, b1, w2, b2, w4, b4, w5, b5, wsel, bsel, wq, bq, c=None, wc=None, y=None, q=0.5):
    """Temporal Fusion Transformer gating and variable selection.

    Temporal Fusion Transformer gating and variable selection.

    Quoted from the paper:
        (2)-(4) "GRN_omega(a, c) = LayerNorm(a + GLU_omega(eta_1))",
                with eta_1 a linear map of eta_2 and
                eta_2 = ELU(W_2 a + W_3 c + b_2)
        (5)  "GLU_omega(gamma)
                 = sigma(W_4,omega gamma + b_4,omega)
                   * (W_5,omega gamma + b_5,omega)"
        (6)  "v_chi_t = Softmax(GRN_v_chi(Xi_t, c_s))"
        (23) "yhat(q, t, tau) = W_q psitilde(t, tau) + b_q"
        (25) "QL(y, yhat, q) = q(y - yhat)_+ + (1 - q)(yhat - y)_+"

    -- Lim, B., Arik, S. O., Loeff, N. and Pfister, T., "Temporal
    Fusion Transformers for Interpretable Multi-horizon Time Series
    Forecasting", International Journal of Forecasting 37(4):1748-1764
    (arXiv:1912.09363).

    All weights are caller-supplied.  ``c`` is the optional static
    context of eq. (3), ``wc`` its projection; omit both for the
    context-free form.  Supplying ``y`` also evaluates eq. (25).

    Parameters
    ----------
    a : as documented for the shelf core
        See ``morie.fn._joseph.tftnet``.
    w1 : as documented for the shelf core
        See ``morie.fn._joseph.tftnet``.
    b1 : as documented for the shelf core
        See ``morie.fn._joseph.tftnet``.
    w2 : as documented for the shelf core
        See ``morie.fn._joseph.tftnet``.
    b2 : as documented for the shelf core
        See ``morie.fn._joseph.tftnet``.
    w4 : as documented for the shelf core
        See ``morie.fn._joseph.tftnet``.
    b4 : as documented for the shelf core
        See ``morie.fn._joseph.tftnet``.
    w5 : as documented for the shelf core
        See ``morie.fn._joseph.tftnet``.
    b5 : as documented for the shelf core
        See ``morie.fn._joseph.tftnet``.
    wsel : as documented for the shelf core
        See ``morie.fn._joseph.tftnet``.
    bsel : as documented for the shelf core
        See ``morie.fn._joseph.tftnet``.
    wq : as documented for the shelf core
        See ``morie.fn._joseph.tftnet``.
    bq : as documented for the shelf core
        See ``morie.fn._joseph.tftnet``.
    c : as documented for the shelf core
        See ``morie.fn._joseph.tftnet``.
    wc : as documented for the shelf core
        See ``morie.fn._joseph.tftnet``.
    y : as documented for the shelf core
        See ``morie.fn._joseph.tftnet``.
    q : as documented for the shelf core
        See ``morie.fn._joseph.tftnet``.

    Returns
    -------
    result : RichResult
        Payload keys: topvar, maxweight, entropy, grnnorm.

    References
    ----------
    Lim, Arik, Loeff and Pfister (2021) IJF 37(4), arXiv:1912.09363, eqs. (2)-(6), (23), (25)
    """
    res = _core.tftnet(a=a, w1=w1, b1=b1, w2=w2, b2=b2, w4=w4, b4=b4, w5=w5, b5=b5, wsel=wsel, bsel=bsel, wq=wq, bq=bq, c=c, wc=wc, y=y, q=q)
    return RichResult(
        title=_METHOD,
        summary_lines=[("topvar", res["topvar"]), ("maxweight", res["maxweight"]), ("entropy", res["entropy"]), ("grnnorm", res["grnnorm"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_temporal_fusion_transformer = tftnet


def cheatsheet():
    return "tftnet: Temporal Fusion Transformer gating and variable selection"
