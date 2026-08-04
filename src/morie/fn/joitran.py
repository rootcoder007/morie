# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""iTransformer attention across variate tokens.

Liu et al. (2024) ICLR, arXiv:2310.06625, eqs. (1)-(2)
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["itrans", "joseph_itransformer"]

_METHOD = "iTransformer attention across variate tokens"


def itrans(x, wembed, bembed, wq, wk, wv, wffn1, bffn1, wffn2, bffn2, wproj, bproj):
    """iTransformer attention across variate tokens.

    iTransformer: variates as tokens, attention across variates.

    Quoted from the paper:
        (1)  "h^0_n = Embedding(X_{:,n});
              H^{l+1} = TrmBlock(H^l), l = 0..L-1;
              Yhat_{:,n} = Projection(h^L_n)"
        (2)  "LayerNorm(H) = {[h_n - Mean(h_n)]/sqrt(Var(h_n))
                              | n = 1..N}"
        attention scores "A_{i,j} = (Q K^T / sqrt(d_k))_{i,j}"

    -- Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L. and Long,
    M., "iTransformer: Inverted Transformers Are Effective for Time
    Series Forecasting", ICLR 2024 (arXiv:2310.06625).

    The inversion is the whole point: each VARIATE series becomes one
    token, so the attention matrix is N x N over variates rather than
    T x T over time steps.  All projections are caller-supplied.

    Parameters
    ----------
    x : as documented for the shelf core
        See ``morie.fn._joseph.itrans``.
    wembed : as documented for the shelf core
        See ``morie.fn._joseph.itrans``.
    bembed : as documented for the shelf core
        See ``morie.fn._joseph.itrans``.
    wq : as documented for the shelf core
        See ``morie.fn._joseph.itrans``.
    wk : as documented for the shelf core
        See ``morie.fn._joseph.itrans``.
    wv : as documented for the shelf core
        See ``morie.fn._joseph.itrans``.
    wffn1 : as documented for the shelf core
        See ``morie.fn._joseph.itrans``.
    bffn1 : as documented for the shelf core
        See ``morie.fn._joseph.itrans``.
    wffn2 : as documented for the shelf core
        See ``morie.fn._joseph.itrans``.
    bffn2 : as documented for the shelf core
        See ``morie.fn._joseph.itrans``.
    wproj : as documented for the shelf core
        See ``morie.fn._joseph.itrans``.
    bproj : as documented for the shelf core
        See ``morie.fn._joseph.itrans``.

    Returns
    -------
    result : RichResult
        Payload keys: nvariates, T, D, attndiag, sumsq.

    References
    ----------
    Liu et al. (2024) ICLR, arXiv:2310.06625, eqs. (1)-(2)
    """
    res = _core.itrans(x=x, wembed=wembed, bembed=bembed, wq=wq, wk=wk, wv=wv, wffn1=wffn1, bffn1=bffn1, wffn2=wffn2, bffn2=bffn2, wproj=wproj, bproj=bproj)
    return RichResult(
        title=_METHOD,
        summary_lines=[("nvariates", res["nvariates"]), ("T", res["T"]), ("D", res["D"]), ("attndiag", res["attndiag"]), ("sumsq", res["sumsq"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_itransformer = itrans


def cheatsheet():
    return "itrans: iTransformer attention across variate tokens"
