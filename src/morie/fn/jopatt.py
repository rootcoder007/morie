# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""PatchTST patching with instance normalization.

Nie, Nguyen, Sinthong and Kalagnanam (2023) ICLR, arXiv:2211.14730, sec. 3.1
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["patchts", "joseph_patchtst"]

_METHOD = "PatchTST patching with instance normalization"


def patchts(x, patchlen, stride, eps=1e-05):
    """PatchTST patching with instance normalization.

    PatchTST patching with reversible instance normalization.

    Quoted from the paper: "the patching process will generate the a
    sequence of patches where N is the number of patches,
    N = floor((L - P)/S) + 2", with "S repeated numbers of the last
    value" padded before patching; each series is normalized to "zero
    mean and unit standard deviation" and the statistics restored on
    output; and channel-independence means a multivariate series is
    "split to M univariate series ... each of them is fed
    independently into the Transformer backbone".

    -- Nie, Y., Nguyen, N. H., Sinthong, P. and Kalagnanam, J., "A Time
    Series is Worth 64 Words: Long-term Forecasting with Transformers",
    ICLR 2023 (arXiv:2211.14730), sec. 3.1.

    ``x`` may be a single series or a list of channels; each channel is
    handled on its own, which IS the channel-independence claim.

    Parameters
    ----------
    x : as documented for the shelf core
        See ``morie.fn._joseph.patchts``.
    patchlen : as documented for the shelf core
        See ``morie.fn._joseph.patchts``.
    stride : as documented for the shelf core
        See ``morie.fn._joseph.patchts``.
    eps : as documented for the shelf core
        See ``morie.fn._joseph.patchts``.

    Returns
    -------
    result : RichResult
        Payload keys: npatches, n, mean, sd, patchsumsq.

    References
    ----------
    Nie, Nguyen, Sinthong and Kalagnanam (2023) ICLR, arXiv:2211.14730, sec. 3.1
    """
    res = _core.patchts(x=x, patchlen=patchlen, stride=stride, eps=eps)
    return RichResult(
        title=_METHOD,
        summary_lines=[("npatches", res["npatches"]), ("n", res["n"]), ("mean", res["mean"]), ("sd", res["sd"]), ("patchsumsq", res["patchsumsq"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_patchtst = patchts


def cheatsheet():
    return "patchts: PatchTST patching with instance normalization"
