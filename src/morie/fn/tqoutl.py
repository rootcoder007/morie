# morie.fn -- function file (rootcoder007/morie)
"""Outlier channel split for KV quantization."""

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["turboquant_outlier_channel_split"]


def turboquant_outlier_channel_split(channels, outlier_threshold=0.99):
    """Separate the few large key channels from the rest.

    The distortion bound is proportional to the embedding norm, and in
    the deeper layers a handful of fixed coordinates carry most of that
    norm.  Splitting them off and quantizing them separately, at a lower
    compression rate, shrinks the norm that the main quantizer has to
    cope with -- which is a much larger win than spending the same bits
    uniformly.  The channels are fixed across tokens, so the split is
    identified once during the prompt phase and then reused.

    Formula: flag channel ``j`` when ``|a_j|`` exceeds the
    ``outlier_threshold`` quantile of the channel magnitudes.

    Parameters
    ----------
    channels : array-like
        Per-channel activation magnitudes (or a matrix, in which case
        the column maxima are used).
    outlier_threshold : float, default 0.99
        Quantile above which a channel is called an outlier.

    Returns
    -------
    RichResult
        ``outlier_idx`` and ``inlier_idx`` (zero-based), ``cut``,
        ``estimate`` (the fraction split off), ``d``.

    References
    ----------
    Zandieh, A., Daliri, M. & Han, I. (2024).  QJL: 1-bit quantized
    JL transform for KV cache quantization with zero overhead.
    arXiv:2406.03482.  Fetched and read; the definitions and bounds used
    here are that paper own (definition 3.1, fact 3.4, lemma 3.5,
    theorem 3.6).  The KV-cache system built on it is Zandieh, A. et al.
    (2025), TurboQuant: online vector quantization with near-optimal
    distortion rate, arXiv:2504.19874.
    """
    Cm = C.mat(channels)
    n, p = C.shape(Cm)
    if p == 1:
        mag = [abs(row[0]) for row in Cm]
    else:
        mag = [max(abs(Cm[i][j]) for i in range(n)) for j in range(p)]
    cut = S.quantile7(mag, float(outlier_threshold))
    out_idx = [j for j in range(len(mag)) if mag[j] > cut]
    in_idx = [j for j in range(len(mag)) if mag[j] <= cut]
    return RichResult(payload={
        "outlier_idx": out_idx, "inlier_idx": in_idx, "cut": cut,
        "estimate": len(out_idx) / len(mag), "d": len(mag),
        "method": "Outlier channel split for KV quantization"})


def cheatsheet():
    return "tqoutl: Outlier channel split for KV quantization."
