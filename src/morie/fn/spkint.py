# morie.fn -- function file (rootcoder007/morie)
"""Spike-train information rate by the direct method."""

from math import fsum, log

from ._richresult import RichResult
from ._spx import vec

__all__ = [
    "spike_information",
    "spikeinfo",
]


def spike_information(spike, stim, nbins=2):
    """Mutual information between a spike response and a stimulus label.

    NOT IN SCHABENBERGER & GOTWAY -- this is neurophysiology. The
    estimator is the "direct method" of Strong, S. P., Koberle, R., de
    Ruyter van Steveninck, R. R. & Bialek, W. (1998), "Entropy and
    information in neural spike trains", *Physical Review Letters*
    80:197-200 -- named from the general literature and NOT verified
    against a PDF in this corpus.

    The information carried about the stimulus is the total response
    entropy minus the entropy that remains once the stimulus is known:

        I = H(R) - H(R|S) = H_total - H_noise,

    in bits (log base 2). H_noise is the STIMULUS-WEIGHTED average of the
    per-stimulus response entropies; weighting the per-stimulus entropies
    equally instead is the standard error and inflates I whenever the
    stimulus classes are unbalanced.

    Responses are discretised into `nbins` equal-count bins (quantiles of
    the observed values), not equal-width bins: equal-width bins on a
    skewed spike-count distribution put nearly every observation in one
    bin and report I close to zero regardless of the truth.

    The direct method is BIASED UPWARD at finite sample size -- every
    spurious response-stimulus coincidence looks like information. No bias
    correction is applied here; the naive value is returned along with
    ``n_per_cell``, the mean occupancy of the response-by-stimulus table,
    because that number is what tells you whether to trust it (rules of
    thumb want it well above 1).

    Parameters
    ----------
    spike : (n,) array-like
        Response values, e.g. spike counts per trial.
    stim : (n,) array-like
        Stimulus class labels, compared as integers.
    nbins : int
        Response bins, at least 2.

    Returns
    -------
    RichResult
        ``information``, ``h_total``, ``h_noise``, ``n_stimuli``,
        ``nbins``, ``n_per_cell``, ``n``, ``method``.
    """
    r = vec(spike, "spike")
    s = vec(stim, "stim")
    n = len(r)
    if len(s) != n:
        raise ValueError("`spike` and `stim` must have the same length")
    if n < 4:
        raise ValueError("at least 4 trials are needed")
    nbins = int(nbins)
    if nbins < 2:
        raise ValueError("`nbins` must be at least 2")
    if nbins > n:
        raise ValueError("`nbins` (%d) exceeds the number of trials (%d)"
                         % (nbins, n))
    si = [int(round(t)) for t in s]
    for t, u in zip(s, si):
        if abs(t - u) > 1e-9:
            raise ValueError("`stim` must hold integer class labels")
    keys = sorted(set(si))
    if len(keys) < 2:
        raise ValueError("at least 2 stimulus classes are needed")

    srt = sorted(r)
    edges = [srt[int(round(n * (b + 1.0) / nbins)) - 1]
             for b in range(nbins - 1)]

    def binof(v):
        for b in range(nbins - 1):
            if v <= edges[b]:
                return b
        return nbins - 1

    code = [binof(t) for t in r]

    def ent(codes):
        m = len(codes)
        h = 0.0
        for b in range(nbins):
            cnt = len([t for t in codes if t == b])
            if cnt:
                p = cnt / float(m)
                h = h - p * log(p, 2.0)
        return h

    htot = ent(code)
    hnoise = 0.0
    for c in keys:
        sub = [code[i] for i in range(n) if si[i] == c]
        hnoise = hnoise + (len(sub) / float(n)) * ent(sub)

    return RichResult(payload={
        "information": htot - hnoise,
        "h_total": htot,
        "h_noise": hnoise,
        "n_stimuli": float(len(keys)),
        "nbins": float(nbins),
        "n_per_cell": n / float(nbins * len(keys)),
        "bits": True,
        "biased_upward_at_small_n": True,
        "equal_count_bins": True,
        "n": n,
        "method": ("Direct-method spike-train information "
                   "(Strong et al. 1998), no bias correction; NOT in "
                   "Schabenberger & Gotway"),
    })


def cheatsheet():
    return "spkint: direct-method spike-train information rate"


# compact alias per ledger/NAMING.md
spikeinfo = spike_information
