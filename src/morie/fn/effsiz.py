# morie.fn -- function file (rootcoder007/morie)
"""Effective sample size under a survey design effect."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["effective_sample_size"]


def effective_sample_size(n, deff=None, icc=None, cluster_size=None):
    r"""Effective sample size for a clustered or weighted survey design.

    .. math::
        n_{\text{eff}} = \frac{n}{\mathrm{deff}},
        \qquad
        \mathrm{deff} = 1 + (\bar m - 1)\,\rho

    for a clustered design with average cluster size :math:`\bar m` and
    intra-class correlation :math:`\rho`.

    This is the survey-sampling notion, not the MCMC one -- see
    :func:`~morie.fn.bayess.effective_sample_size_bayes` for that. Both answer
    "how many independent observations is this worth", but from different
    sources of dependence.

    The consequence people underestimate is how fast clustering bites. With
    :math:`\rho = 0.05` -- modest by the standards of school, clinic or
    neighbourhood data -- clusters of 50 give
    :math:`\mathrm{deff} = 3.45`, so a survey of 5000 carries the precision of
    1450. Treating it as 5000 understates every standard error by 86%.

    Parameters
    ----------
    n : int
        Nominal sample size.
    deff : float, optional
        Design effect, if known directly.
    icc : float, optional
        Intra-class correlation, with ``cluster_size``.
    cluster_size : float, optional
        Average cluster size.

    Returns
    -------
    RichResult
        ``n_effective``, ``deff``, ``se_inflation``, ``information_lost``.

    References
    ----------
    Kish, L. (1965). *Survey Sampling*. Wiley.

    Examples
    --------
    Modest clustering costs a great deal of information.

    >>> r = effective_sample_size(5000, icc=0.05, cluster_size=50)
    >>> float(round(r["deff"], 2))
    3.45
    >>> int(round(r["n_effective"]))
    1449

    Standard errors inflate by the square root of the design effect.

    >>> bool(abs(r["se_inflation"] - 3.45 ** 0.5) < 1e-9)
    True

    No clustering means no loss.

    >>> float(effective_sample_size(1000, icc=0.0, cluster_size=20)["deff"])
    1.0

    >>> effective_sample_size(100)
    Traceback (most recent call last):
        ...
    ValueError: supply deff, or both icc and cluster_size
    """
    n = int(n)
    if n < 1:
        raise ValueError("n must be at least 1")
    if deff is None:
        if icc is None or cluster_size is None:
            raise ValueError("supply deff, or both icc and cluster_size")
        if not 0.0 <= icc <= 1.0:
            raise ValueError("icc must be in [0, 1]")
        if cluster_size < 1:
            raise ValueError("cluster_size must be at least 1")
        deff = 1.0 + (float(cluster_size) - 1.0) * float(icc)
    deff = float(deff)
    if deff <= 0:
        raise ValueError("deff must be positive")
    n_eff = n / deff
    return RichResult(
        title="Effective sample size (design)",
        summary_lines=[("n", n), ("deff", deff), ("n_eff", float(n_eff))],
        warnings=([f"the design effect is {deff:.2f}, so treating n as {n} "
                   f"would understate standard errors by "
                   f"{100 * (np.sqrt(deff) - 1):.0f}%"] if deff > 1.5 else []),
        payload={
            "n_effective": float(n_eff), "deff": deff,
            "se_inflation": float(np.sqrt(deff)),
            "information_lost": float(1.0 - 1.0 / deff),
            "n": n, "icc": icc, "cluster_size": cluster_size,
            "method": "effective_sample_size",
        },
    )


def cheatsheet():
    return "effsiz: SURVEY deff, not MCMC ESS; icc 0.05 with clusters of 50 costs 70% of your sample"
