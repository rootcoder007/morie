"""Bootstrap-calibrated confidence interval (Loh 1991)."""

import math

from . import _array_core as np
from ._stats_core import norm
from ._richresult import RichResult

__all__ = ["btcalib", "bootstrap_calibration_ci"]


def btcalib(x, alpha=0.05, B=1000, seed=0):
    """
    Exact bootstrap calibration of the normal-theory mean interval.

    Loh (1991), Sec. 2.1 ("Exact calibration"): for each bootstrap
    sample X*_i compute the t statistic
    t*_i = n^{1/2}(theta*_i - theta_hat)/sigma*_i and, for the
    two-sided 100(1 - 2a)% interval, beta_hat_i = 1 - Phi(|t*_i|)
    (his Eq. 2); the calibrated nominal level alpha' is the
    2a-quantile of {beta_hat_i}, and the interval is the
    normal-theory interval run at alpha' instead of a.  His Sec.
    2.2 proves the shortcut z_{1-alpha'} = (1 - 2a)-quantile of
    {|t*_i|}, so the calibrated interval IS Beran's bootstrap-root
    (Efron's bootstrap-t) interval -- an identity this
    implementation verifies by computing both routes.

    Sources
    -------
    Loh, W.-Y. (1991). Bootstrap calibration for confidence
    interval construction and selection. *Statistica Sinica*, 1(2),
    477-491, Sec. 2.1 Eqs. 1-2 and Sec. 2.2 (local copy
    fetched-wave3/Bootstrap_calibration_for_confidence_interval_
    construction_and_selection..pdf; read from the rendered scan).

    Parameters
    ----------
    x : sequence of float
        Sample.
    alpha : float
        Total two-sided non-coverage 2a (0.05 = 95% interval).
    B : int
        Bootstrap samples.
    seed : int
        Native-RNG seed (SplitMix64; mirrored by .ghc_rng in R).

    Returns
    -------
    RichResult
        Keys: estimate, lower, upper, alpha_prime, z_calibrated,
        identity_gap (|calibrated - bootstrap-t| endpoint gap).
    """
    xv = [float(v) for v in x]
    n = len(xv)
    if n < 5:
        raise ValueError("need at least five observations")
    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    a = alpha / 2.0
    rng = np.random.default_rng(seed)
    that = sum(xv) / n
    sig = math.sqrt(sum((v - that) ** 2 for v in xv) / (n - 1))
    sqn = math.sqrt(n)
    tstars = []
    betas = []
    for _b in range(int(B)):
        xb = [xv[min(int(float(rng.uniform()) * n), n - 1)]
              for _ in range(n)]
        mb = sum(xb) / n
        sb = math.sqrt(sum((v - mb) ** 2 for v in xb) / (n - 1))
        if sb <= 0:
            sb = 1e-300
        t = sqn * (mb - that) / sb
        tstars.append(abs(t))
        betas.append(1.0 - float(norm.cdf(abs(t))))       # Loh Eq. 2
    # alpha' = 2a-quantile of the beta_hat_i (Sec. 2.1)
    sb_ = sorted(betas)
    idx = max(min(int(math.ceil(2.0 * a * B)) - 1, B - 1), 0)
    alpha_prime = sb_[idx]
    z_cal = float(norm.ppf(1.0 - alpha_prime))
    half_cal = z_cal * sig / sqn
    # Sec. 2.2 shortcut: z_{1-alpha'} = (1-2a)-quantile of |t*|; the
    # SAME order statistic as the beta quantile (beta is monotone
    # decreasing in |t*|), i.e. ascending index B - 1 - idx
    st = sorted(tstars)
    z_boot = st[B - 1 - idx]
    half_bt = z_boot * sig / sqn
    gap = abs(half_cal - half_bt)
    return RichResult(payload={
        "estimate": that,
        "lower": that - half_cal,
        "upper": that + half_cal,
        "alpha_prime": alpha_prime,
        "z_calibrated": z_cal,
        "identity_gap": gap,
        "alpha": alpha,
        "B": int(B),
        "seed": int(seed),
        "method": "Loh (1991) exact bootstrap calibration (Eqs. 1-2)",
    })


# long descriptive alias (stub-era name)
bootstrap_calibration_ci = btcalib


def cheatsheet():
    return "btcalib: beta_i = 1 - Phi(|t*_i|); alpha' = q_{2a}(beta); == bootstrap-t"

# public names resolved by fn/_lazy_map.json
boot_calibrated_ci = btcalib
