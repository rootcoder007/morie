# SPDX-License-Identifier: AGPL-3.0-or-later
"""ChIP-seq peak significance with the MACS dynamic Poisson lambda."""

import math

from . import _array_core as np

from ._richresult import RichResult
from . import _sci_core as sci

__all__ = ["chipsq", "chip_seq_peak"]


def _pois_upper(k, lam):
    # P(X >= k) for X ~ Poisson(lam); equals the regularized lower
    # incomplete gamma P(k, lam) for integer k >= 1.
    if k <= 0:
        return 1.0
    if lam <= 0.0:
        return 0.0
    return float(sci.gammainc(float(k), float(lam)))


def chipsq(count, width, lambda_bg, count_1k=None, count_5k=None,
           count_10k=None, use_1k=True):
    """
    Candidate ChIP-seq peak significance, MACS dynamic lambda.

    For each candidate peak of a given width with k ChIP tags, MACS
    replaces the uniform genome background rate lambda_BG by the
    dynamic local parameter

        lambda_local = max(lambda_BG, [lambda_1k,] lambda_5k,
                           lambda_10k),

    where lambda_Xk is estimated from the tag count of the X-kb
    window centred on the peak in the control sample (or the ChIP
    sample itself when no control exists, in which case lambda_1k is
    omitted -- pass use_1k=False). Each window count is rescaled to
    the peak width: lambda_Xk = count_Xk * width / X000. The peak
    p-value is the Poisson upper tail P(X >= k | lambda_local), and
    the reported fold_enrichment is k / lambda_local.

    Parameters
    ----------
    count : int or array-like of int
        ChIP tag count k in each candidate peak region.
    width : float or array-like
        Peak region width in bp.
    lambda_bg : float
        Expected background tag count for a region of this width
        (genome-wide rate times width).
    count_1k, count_5k, count_10k : int or array-like, optional
        Control tag counts in the centred 1 kb, 5 kb, 10 kb windows.
        Any omitted window is skipped in the maximum.
    use_1k : bool
        Set False when no control sample exists (MACS then drops
        lambda_1k).

    Returns
    -------
    result : RichResult
        Keys: pvalue, lambda_local, fold_enrichment, count, width,
        n_peaks, method.

    References
    ----------
    Zhang, Y., Liu, T., Meyer, C. A., Eeckhoute, J., Johnson, D. S.,
    Bernstein, B. E., Nusbaum, C., Myers, R. M., Brown, M., Li, W.
    and Liu, X. S. (2008), "Model-based Analysis of ChIP-Seq
    (MACS)", Genome Biology 9(9), R137. Dynamic
    lambda_local = max(lambda_BG, [lambda_1k,] lambda_5k, lambda_10k)
    and Poisson p-value / fold_enrichment from the section "Methods,
    Peak detection". Local source:
    library/pdf/fetched-wave3/Zhang-2008-MACS-GenomeBiology.pdf.
    """
    k = np.atleast_1d(np.asarray(count, dtype=float))
    npk = len(k)

    def _vec(x):
        if x is None:
            return None
        v = np.atleast_1d(np.asarray(x, dtype=float))
        if len(v) == 1 and npk > 1:
            v = np.asarray([float(v[0])] * npk)
        if len(v) != npk:
            raise ValueError("window counts must match count length")
        return v

    wv = _vec(width)
    c1 = _vec(count_1k)
    c5 = _vec(count_5k)
    c10 = _vec(count_10k)
    lam_loc = []
    pv = []
    fe = []
    for i in range(npk):
        lam = float(lambda_bg)
        if c1 is not None and use_1k:
            lam = max(lam, float(c1[i]) * float(wv[i]) / 1000.0)
        if c5 is not None:
            lam = max(lam, float(c5[i]) * float(wv[i]) / 5000.0)
        if c10 is not None:
            lam = max(lam, float(c10[i]) * float(wv[i]) / 10000.0)
        lam_loc.append(lam)
        pv.append(_pois_upper(int(k[i]), lam))
        fe.append(float(k[i]) / lam if lam > 0.0 else float("inf"))
    return RichResult(payload={
        "pvalue": np.asarray(pv),
        "lambda_local": np.asarray(lam_loc),
        "fold_enrichment": np.asarray(fe),
        "count": k,
        "width": wv,
        "n_peaks": npk,
        "method": "MACS dynamic-lambda Poisson peak test (Zhang et al. 2008)",
    })


chip_seq_peak = chipsq
chipseqpeak = chipsq


def cheatsheet():
    return ("chipsq(count, width, lambda_bg, count_1k, count_5k, "
            "count_10k) -> MACS lambda_local Poisson peak p-values.")
