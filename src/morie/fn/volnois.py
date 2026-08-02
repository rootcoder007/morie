# morie.fn -- function file (rootcoder007/morie)
"""Microstructure noise variance from high-frequency returns."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["vol_noise_variance", "vol_noise_variance_est"]


def vol_noise_variance(r_intraday, K=None):
    r"""Estimate the market-microstructure noise variance from
    ultra-high-frequency returns (Zhang, Mykland and Ait-Sahalia
    2005; Ait-Sahalia, Mykland and Zhang 2005).

    With observed log-price = efficient price + i.i.d. noise
    :math:`\varepsilon`, the realized variance at the finest grid
    does NOT estimate integrated variance -- it diverges linearly:

    .. math:: E[RV^{(all)}] = IV + 2n\,E[\varepsilon^2],

    so at high frequency the noise term dominates and

    .. math:: \widehat{E[\varepsilon^2]} = \frac{RV^{(all)}}{2n}

    (AMZ 2005, Eq. (3.7) territory). That divergence is the famous
    volatility "signature plot", and it is why naive RV gets WORSE as
    data get better. The two-scale estimator of ZMA (2005) then
    recovers the integrated variance itself:

    .. math:: \widehat{IV}^{(TS)} = RV^{(avg,K)}
              - \frac{\bar n}{n} RV^{(all)},

    the K-subsample average RV debiased by the fine-grid RV. Both are
    returned: the noise variance, and the two-scale IV alongside the
    naive RV so the size of the correction is visible.

    Parameters
    ----------
    r_intraday : array-like
        Intraday log-returns at the finest sampling.
    K : int, optional
        Subsampling factor for the two-scale estimate;
        ``max(2, round(n**(2/3)))`` when omitted, the ZMA rate.

    Returns
    -------
    RichResult
        keys: ``noise_variance``, ``noise_sd``, ``rv_all``,
        ``rv_subsampled``, ``iv_two_scale``, ``K``,
        ``noise_share_of_rv``, ``signature_note``, ``n``, ``method``.

    References
    ----------
    Zhang, L., Mykland, P. A. and Ait-Sahalia, Y. (2005), "A tale of
    two time scales", *JASA* 100:1394-1411. Ait-Sahalia, Y., Mykland,
    P. A. and Zhang, L. (2005), "How often to sample a
    continuous-time process in the presence of market microstructure
    noise", *Review of Financial Studies* 18:351-416.
    """
    r = np.asarray(r_intraday, dtype=float).ravel()
    n = r.size
    if n < 30:
        raise ValueError(f"need at least 30 intraday returns, got {n}.")
    rv_all = float(np.sum(r ** 2))
    noise_var = rv_all / (2.0 * n)
    KK = max(2, int(round(n ** (2.0 / 3.0)))) if K is None else int(K)
    if not 2 <= KK <= n // 2:
        raise ValueError(f"K must lie in 2..{n // 2}, got {KK}.")
    # K-subsample average RV: prices at every K-th tick, K offsets
    p = np.concatenate([[0.0], np.cumsum(r)])
    rvs = []
    counts = []
    for off in range(KK):
        sub = p[off::KK]
        if sub.size >= 2:
            rvs.append(float(np.sum(np.diff(sub) ** 2)))
            counts.append(sub.size - 1)
    rv_avg = float(np.mean(rvs))
    nbar = float(np.mean(counts))
    iv_ts = rv_avg - (nbar / n) * rv_all
    return RichResult(payload={
        "noise_variance": noise_var,
        "noise_sd": float(np.sqrt(noise_var)),
        "rv_all": rv_all, "rv_subsampled": rv_avg,
        "iv_two_scale": iv_ts, "K": int(KK),
        "noise_share_of_rv": float(2 * n * noise_var / rv_all),
        "signature_note": "E[RV_all] = IV + 2n E[eps^2]: at the finest grid "
                          "the noise term dominates, which is why naive RV "
                          "gets WORSE as sampling gets finer",
        "ts_note": "the two-scale IV debiases the subsampled RV with the "
                   "fine-grid RV (ZMA 2005); a negative value at small n "
                   "means noise swamps signal at this K",
        "n": int(n),
        "method": "Noise variance RV_all/(2n) and two-scale IV (ZMA 2005; AMZ 2005)"})


def cheatsheet():
    return "volnois: RV diverges as 2n eps^2 -- the divergence IS the noise estimate, TSRV recovers IV"


#: Catalogue alias for :func:`vol_noise_variance`.
vol_noise_variance_est = vol_noise_variance
