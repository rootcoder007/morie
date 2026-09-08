r"""Robust confidence intervals for RD designs (front end).

Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014) "Robust Nonparametric
Confidence Intervals for Regression-Discontinuity Designs", *Econometrica*
82(6), 2295-2326.

This module is the *interval* half of the paper and shares one implementation
with :mod:`morie.fn.causrddc`, which carries the estimator, the MSE-optimal
bandwidths of Lemma 1 and the full documentation. Nothing is reimplemented
here -- two ledger entries point at the same paper, and duplicating the
numerics would give two arms that could silently drift apart.

What this front end adds is the comparison the paper is about, in one table:
for a single fit it returns all three intervals side by side, together with
the coverage-relevant quantities that distinguish them.

.. math::

   I_{\mathrm{SRD}}(h_n) &= \hat\tau(h_n)
       \pm \Phi^{-1}_{1-\alpha/2}\sqrt{V(h_n)} \\
   I^{bc}_{\mathrm{SRD}}(h_n, b_n) &= \hat\tau(h_n) - h_n^{p+1-\nu}\hat B
       \pm \Phi^{-1}_{1-\alpha/2}\sqrt{V(h_n)} \\
   I^{rbc}_{\mathrm{SRD}}(h_n, b_n) &= \hat\tau(h_n) - h_n^{p+1-\nu}\hat B
       \pm \Phi^{-1}_{1-\alpha/2}\sqrt{V(h_n) + C^{bc}(h_n, b_n)}

The first undercovers at an MSE-optimal bandwidth because the leading bias
does not vanish; the second recentres but keeps a variance that ignores the
bias estimate's own variability; only the third does both, which is what
Theorem 1 buys and what Remark 2 calls robustness to "small" or "large"
bandwidths.

See :mod:`morie.fn.causrddc` for the estimator, kernels, variance routes
(nearest-neighbour or plug-in residuals), fuzzy and kink designs, and the
bandwidth selectors.
"""

from .causrddc import causrddc

from ._richresult import RichResult

__all__ = ["rdrobu", "calonico_cattaneo_titiunik", "rd_confidence_intervals"]


def rdrobu(y, x, cutoff=0.0, alpha=0.05, **kwargs):
    r"""The three RD confidence intervals of Calonico et al. (2014).

    Parameters
    ----------
    y, x : array-like
        Outcome and running variable.
    cutoff : float
        The threshold; ``x >= cutoff`` is treated.
    alpha : float
        1 - coverage.
    **kwargs
        Passed through to :func:`morie.fn.causrddc.causrddc`: ``p``, ``q``,
        ``nu``, ``h``, ``b``, ``kernel``, ``vce``, ``J``, ``treatment``,
        ``contrast``.

    Returns
    -------
    RichResult
        ``estimate`` is the conventional point estimate and
        ``bias_corrected`` the recentred one; ``intervals`` is a dict with
        ``conventional``, ``bias_corrected`` and ``robust``; ``widths``
        their lengths; ``se_conventional`` and ``se_robust`` the two
        standard errors; ``correction_factor`` is
        :math:`\sqrt{(V + C^{bc})/V}`, the ratio by which accounting for the
        bias estimate's variability widens the interval; plus ``h``, ``b``,
        ``rho`` and the rest of the fit.

    Examples
    --------
    ::

        r = rdrobu(y, x)
        r["intervals"]["robust"], r["correction_factor"]

    References
    ----------
    Calonico, Cattaneo & Titiunik (2014) *Econometrica* 82(6), 2295-2326,
    Theorem 1 and Remarks 2-5.
    """
    fit = causrddc(y, x, cutoff=cutoff, alpha=alpha, **kwargs)
    ci_c = fit["ci_conventional"]
    ci_b = fit["ci_bias_corrected"]
    ci_r = fit["ci_robust"]
    se_c = fit["se_conventional"]
    se_r = fit["se_robust"]
    payload = dict(fit)
    payload.update({
        "estimate": fit["estimate"],
        "intervals": {"conventional": ci_c, "bias_corrected": ci_b,
                      "robust": ci_r},
        "widths": {"conventional": ci_c[1] - ci_c[0],
                   "bias_corrected": ci_b[1] - ci_b[0],
                   "robust": ci_r[1] - ci_r[0]},
        "correction_factor": (se_r / se_c) if se_c > 0 else float("nan"),
        "bias_estimate": fit["estimate"] - fit["bias_corrected"],
        "method": "robust bias-corrected RD confidence intervals "
                  "(Calonico, Cattaneo & Titiunik 2014)",
        "note": "one implementation, shared with morie.fn.causrddc; the "
                "conventional interval is the one the paper shows to "
                "undercover at an MSE-optimal bandwidth",
    })
    return RichResult(payload=payload)


def calonico_cattaneo_titiunik(y, x, cutoff=0.0, **kwargs):
    """Alias kept from the generated stub's signature."""
    return rdrobu(y, x, cutoff, **kwargs)


def cheatsheet():
    return ("rdrobu: the three RD intervals of Calonico, Cattaneo & "
            "Titiunik (2014) side by side -- conventional, bias-corrected, "
            "and robust (recentred AND rescaled by V + C^bc). Shares its "
            "implementation with causrddc; see that module for the "
            "estimator, bandwidths and designs.")


# compact alias per ledger/NAMING.md
rd_confidence_intervals = rdrobu
