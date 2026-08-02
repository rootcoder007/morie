# morie.fn -- function file (rootcoder007/morie)
"""Pitman efficiency by simulation."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["gibbons_pitman_efficiency"]


def gibbons_pitman_efficiency(test1, test2, sampler, theta0=0.0, delta=0.5,
                              n=200, n_sim=400, alpha=0.05, seed=0):
    r"""Finite-sample estimate of the Pitman efficiency idea
    (Gibbons Ch. 1.2.11): the ratio of sample sizes two tests need
    for the same power at the same alternative. Estimated here by
    measuring both tests' power at (n, theta0 + delta) and converting
    through the normal power curve

    .. math:: n_2/n_1 \approx
              \left[\frac{z_\alpha + z_{\beta_1}}
              {z_\alpha + z_{\beta_2}}\right]^2,

    an approximation, and stated as one -- the exact limit needs the
    efficacies (Theorem 13.2.1 route, morie.fn.gb1321).

    Parameters
    ----------
    test1, test2 : callable
        Each maps a sample to a p-value.
    sampler : callable
        sampler(theta, n, rng) -> sample.
    theta0 : float
        Null value.
    delta : float
        Alternative offset.
    n : int
        Sample size per replication.
    n_sim : int
        Replications.
    alpha : float
        Level.
    seed : int
        RNG seed.

    Returns
    -------
    RichResult
        keys: ``efficiency_ratio``, ``power1``, ``power2``, ``n``,
        ``n_sim``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 1.2.11.
    """
    from scipy import stats as st

    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}.")
    n, n_sim = int(n), int(n_sim)
    if n < 5 or n_sim < 20:
        raise ValueError("need n >= 5 and n_sim >= 20.")
    rng = np.random.default_rng(seed)
    rej1 = rej2 = 0
    for _ in range(n_sim):
        x = sampler(theta0 + delta, n, rng)
        rej1 += test1(x) < alpha
        rej2 += test2(x) < alpha
    p1, p2 = rej1 / n_sim, rej2 / n_sim
    za = st.norm.ppf(1 - alpha)
    zb1 = st.norm.ppf(np.clip(p1, 1e-6, 1 - 1e-6))
    zb2 = st.norm.ppf(np.clip(p2, 1e-6, 1 - 1e-6))
    ratio = ((za + zb1) / (za + zb2)) ** 2
    return RichResult(
        payload={
            "efficiency_ratio": float(ratio), "power1": float(p1),
            "power2": float(p2), "n": n, "n_sim": n_sim,
            "method": "Simulated Pitman-type efficiency via the power curve "
                      "(approximation; exact route is the efficacy ratio)",
        }
    )


def cheatsheet():
    return "gb_psi: simulated n2/n1 via normal power curve; approximate by design"
