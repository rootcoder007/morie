# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""TiDE dense encoder-decoder.

Das, Kong, Leach, Mathur, Sen and Yu (2023) TMLR, arXiv:2304.08424, eqs. (3)-(6)
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["tide", "joseph_tide_encoder"]

_METHOD = "TiDE dense encoder-decoder"


def tide(y, feats, fproj, enc, dec, tdec, wglobal, horizon):
    """TiDE dense encoder-decoder.

    TiDE dense encoder-decoder.

    Quoted from the paper:
        (3)  "xtilde_{i,t} = ResidualBlock(x_{i,t})"
        (4)  "e^(i) = Encoder(y^i_{1:L}; xtilde^i_{1:L+H}; a^(i))"
        (5)  "g^(i) = Decoder(e^(i)) in R^{p.H}"
        (6)  "D^(i) = Reshape(g^(i)) in R^{p x H}"
             "yhat^i_{L+t} = TemporalDecoder(d_{i,t}; xtilde^i_{L+t})"

    -- Das, A., Kong, W., Leach, A., Mathur, S., Sen, R. and Yu, R.,
    "Long-term Forecasting with TiDE: Time-series Dense Encoder", TMLR
    2023 (arXiv:2304.08424).  The paper states the residual block and
    the global linear residual connection in prose rather than as
    numbered equations, so those two are implemented from the prose and
    said to be so here.

    ``fproj``, ``enc``, ``dec`` and ``tdec`` are each a caller-supplied
    (w1, b1, w2, b2, wskip) residual block; ``wglobal`` is the global
    linear map from the lookback straight to the horizon, which the
    paper adds to the output.  ``tdec`` must produce one value per
    horizon step.

    Parameters
    ----------
    y : as documented for the shelf core
        See ``morie.fn._joseph.tide``.
    feats : as documented for the shelf core
        See ``morie.fn._joseph.tide``.
    fproj : as documented for the shelf core
        See ``morie.fn._joseph.tide``.
    enc : as documented for the shelf core
        See ``morie.fn._joseph.tide``.
    dec : as documented for the shelf core
        See ``morie.fn._joseph.tide``.
    tdec : as documented for the shelf core
        See ``morie.fn._joseph.tide``.
    wglobal : as documented for the shelf core
        See ``morie.fn._joseph.tide``.
    horizon : as documented for the shelf core
        See ``morie.fn._joseph.tide``.

    Returns
    -------
    result : RichResult
        Payload keys: horizon, p, encdim, first, mean.

    References
    ----------
    Das, Kong, Leach, Mathur, Sen and Yu (2023) TMLR, arXiv:2304.08424, eqs. (3)-(6)
    """
    res = _core.tide(y=y, feats=feats, fproj=fproj, enc=enc, dec=dec, tdec=tdec, wglobal=wglobal, horizon=horizon)
    return RichResult(
        title=_METHOD,
        summary_lines=[("horizon", res["horizon"]), ("p", res["p"]), ("encdim", res["encdim"]), ("first", res["first"]), ("mean", res["mean"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_tide_encoder = tide


def cheatsheet():
    return "tide: TiDE dense encoder-decoder"
