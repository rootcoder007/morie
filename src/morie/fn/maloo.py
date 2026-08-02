# morie.fn -- function file (rootcoder007/morie)
"""Leave-one-out influence diagnostics for meta-analysis."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ma_leave_one_out"]


def ma_leave_one_out(yi, vi, method="PM"):
    r"""Leave-one-out sensitivity for a random-effects pooled
    estimate (Viechtbauer and Cheung 2010): refit the model omitting
    each study in turn and report how far the pooled effect, the
    heterogeneity and the interval move.

    Refitting is the point. Deleting a study changes not only the
    weighted mean but :math:`\tau^2` itself, and therefore ALL the
    weights -- so a one-study deletion can move the pooled estimate
    far more than that study's own weight suggests. Recomputing the
    pooled effect with the full-data :math:`\tau^2` and one study
    dropped, which is the quick version people write, misses exactly
    that channel; ``tau2_loo`` is returned so the effect is visible.

    The diagnostic worth reading is not the largest shift in
    isolation but whether any single deletion changes a CONCLUSION:
    ``flips_significance`` records whether the interval's exclusion
    of zero changes, which is the question a sensitivity analysis is
    usually asked to answer.

    Parameters
    ----------
    yi, vi : array-like
        Effects and within-study variances.
    method : {"PM", "REML", "DL"}
        Heterogeneity estimator used for each refit; the choice
        propagates into every leave-one-out estimate.

    Returns
    -------
    RichResult
        keys: ``mu_full``, ``tau2_full``, ``mu_loo``, ``tau2_loo``,
        ``delta_mu``, ``ci_loo``, ``flips_significance``,
        ``most_influential``, ``significant_full``, ``method_used``,
        ``k``, ``method``.

    References
    ----------
    Viechtbauer, W. and Cheung, M. W.-L. (2010), "Outlier and
    influence diagnostics for meta-analysis", *Research Synthesis
    Methods* 1:112-125. Viechtbauer, W. (2010), *JSS* 36(3).
    """
    from . import _stats_core as stats

    from ._psycho import dersimonian_laird

    y = np.asarray(yi, dtype=float).ravel()
    v = np.asarray(vi, dtype=float).ravel()
    if y.size != v.size:
        raise ValueError(f"yi has {y.size} entries and vi has {v.size}.")
    k = y.size
    if k < 3:
        raise ValueError(f"leave-one-out needs at least 3 studies, got {k}.")
    if np.any(v <= 0):
        raise ValueError("every within-study variance must be positive.")
    if method not in ("PM", "REML", "DL"):
        raise ValueError("method must be 'PM', 'REML' or 'DL'.")

    def fit(yy, vv):
        if method == "DL":
            t2 = dersimonian_laird(yy, vv)
        elif method == "PM":
            from .mapaule import ma_paule_mandel
            t2 = ma_paule_mandel(yy, vv)["tau2"]
        else:
            from .mareml import ma_random_reml
            t2 = ma_random_reml(yy, vv)["tau2"]
        w = 1.0 / (vv + t2)
        mu = float(np.sum(w * yy) / np.sum(w))
        se = float(np.sqrt(1.0 / np.sum(w)))
        return mu, se, float(t2)

    mu_f, se_f, t2_f = fit(y, v)
    z = stats.norm.ppf(0.975)
    sig_full = bool(abs(mu_f) > z * se_f)
    mu_l = np.empty(k)
    t2_l = np.empty(k)
    ci_l = np.empty((k, 2))
    flips = np.zeros(k, dtype=bool)
    idx = np.arange(k)
    for i in range(k):
        keep = idx != i
        mu_i, se_i, t2_i = fit(y[keep], v[keep])
        mu_l[i] = mu_i
        t2_l[i] = t2_i
        ci_l[i] = (mu_i - z * se_i, mu_i + z * se_i)
        flips[i] = bool(abs(mu_i) > z * se_i) != sig_full
    d = mu_l - mu_f
    return RichResult(payload={
        "mu_full": mu_f, "tau2_full": t2_f,
        "mu_loo": mu_l, "tau2_loo": t2_l, "ci_loo": ci_l,
        "delta_mu": d, "flips_significance": flips,
        "most_influential": int(np.argmax(np.abs(d))),
        "max_abs_delta": float(np.max(np.abs(d))),
        "significant_full": sig_full,
        "refit_note": "each fit re-estimates tau^2, so deleting a study "
                      "changes ALL the weights -- recomputing with the "
                      "full-data tau^2 misses that channel",
        "what_to_read": "not the largest shift in isolation but whether any "
                        "single deletion changes a CONCLUSION, which "
                        "flips_significance records",
        "method_used": method, "k": int(k),
        "method": "Leave-one-out influence for random-effects meta-analysis "
                  "(Viechtbauer-Cheung 2010)"})


def cheatsheet():
    return "maloo: refit tau^2 each time -- deletion moves every weight, not just one"
