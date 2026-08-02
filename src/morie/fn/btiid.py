# morie.fn -- function file (rootcoder007/morie)
"""Nonparametric IID bootstrap of a statistic."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["boot_iid_resample"]


def boot_iid_resample(x, stat, B=1000, seed=0):
    r"""Efron's (1979) nonparametric bootstrap: draw :math:`B`
    samples of size :math:`n` WITH replacement from the data, apply
    the statistic to each, and read the sampling distribution off the
    replicates.

    What the replicates estimate is the distribution of the statistic
    under sampling from the EMPIRICAL distribution :math:`\hat F_n`,
    not from :math:`F` -- the bootstrap principle is that the two are
    close when :math:`\hat F_n` is close to :math:`F` in the sense
    the statistic cares about. Where that fails, it fails without
    warning: extremes are the canonical case, since the bootstrap
    distribution of :math:`\max X_i` puts mass
    :math:`1-(1-1/n)^n \to 0.632` on the sample maximum itself and is
    inconsistent. The output says so rather than pretending
    universality.

    The computation is shared with
    ``morie.fn._wsm.bootstrap_replicates`` -- one resampler across
    the bootstrap shelf, no drift.

    Parameters
    ----------
    x : array-like
        Sample; rows are observations.
    stat : callable
        The statistic, applied to a resample.
    B : int, default 1000
        Replicates.
    seed : int, default 0
        Resampling seed.

    Returns
    -------
    RichResult
        keys: ``replicates``, ``estimate`` (on the original data),
        ``se``, ``bias``, ``ci_percentile``, ``B``, ``n``,
        ``consistency_caveat``, ``method``.

    References
    ----------
    Efron, B. (1979), "Bootstrap methods: another look at the
    jackknife", *Annals of Statistics* 7:1-26.
    """
    from ._wsm import bootstrap_replicates

    d = np.asarray(x, dtype=float)
    reps = bootstrap_replicates(d, stat, B=B, seed=seed)
    est = float(stat(d))
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return RichResult(payload={
        "replicates": reps, "estimate": est,
        "se": float(np.std(reps, ddof=1)),
        "bias": float(np.mean(reps) - est),
        "ci_percentile": (float(lo), float(hi)),
        "B": int(reps.size), "n": int(d.shape[0]),
        "consistency_caveat": "the bootstrap estimates the statistic's "
                              "distribution under the EMPIRICAL law; for "
                              "statistics it is inconsistent for -- the "
                              "sample maximum above all -- it fails without "
                              "warning",
        "method": "Efron (1979) nonparametric IID bootstrap"})


def cheatsheet():
    return "btiid: resamples estimate the law under F_n -- and for the max, that is the wrong law"
