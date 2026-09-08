# morie.fn -- function file (rootcoder007/morie)
"""Empirical epsilon of a mechanism on two neighbouring datasets."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["epsilon_dp"]


def epsilon_dp(mech, D, D_prime, n_samples=20000, bins=50, seed=None):
    r"""Estimate the privacy loss a mechanism exhibits between two datasets.

    Pure :math:`\varepsilon`-DP requires, for every output set :math:`S`,

    .. math::
        \frac{P(M(D) \in S)}{P(M(D') \in S)} \le e^{\varepsilon}.

    This samples the mechanism on both datasets, bins the outputs, and returns
    the largest observed log-ratio.

    Read it only as a **lower bound**, and only as evidence of a violation:
    a large empirical epsilon on one pair of datasets disproves a claimed
    guarantee, but a small one proves nothing. The definition quantifies over
    *all* neighbouring pairs and all output sets, and no finite sample can
    check that. This is a debugging instrument -- for catching an
    implementation that forgot its noise, or calibrated it to the wrong
    sensitivity -- not a certificate.

    Sampling error alone inflates the estimate in sparsely populated bins, so
    bins with too few samples in either arm are excluded and counted.

    Parameters
    ----------
    mech : callable
        ``mech(D, rng) -> float`` or ``-> array``. Must be randomised.
    D, D_prime : array-like
        Neighbouring datasets, differing in one record.
    n_samples : int
        Draws per dataset.
    bins : int
        Histogram bins for the output distributions.
    seed : int, optional
        Seed.

    Returns
    -------
    RichResult
        ``epsilon_empirical``, ``max_log_ratio``, ``n_usable_bins``,
        ``n_excluded_bins``.

    References
    ----------
    Ding, Z., Wang, Y., Wang, G., Zhang, D., & Kifer, D. (2018). Detecting
        violations of differential privacy. *CCS 2018*, 475-489.
    Dwork, C., & Roth, A. (2014). The algorithmic foundations of
        differential privacy. *FnT-TCS*, 9(3-4), 211-487.

    Examples
    --------
    A correctly calibrated Laplace mechanism shows an empirical epsilon near
    its nominal one, not far above it.

    >>> import numpy as np
    >>> D, Dp = np.zeros(10), np.r_[np.zeros(9), 1.0]
    >>> lap = lambda X, rng: float(X.sum() + rng.laplace(0, 1 / 0.5))
    >>> r = epsilon_dp(lap, D, Dp, n_samples=40000, seed=0)
    >>> bool(r["epsilon_empirical"] < 1.5)
    True

    A mechanism that adds no noise is caught immediately -- the empirical loss
    is enormous.

    >>> none = lambda X, rng: float(X.sum())
    >>> bool(epsilon_dp(none, D, Dp, n_samples=4000, seed=0)["epsilon_empirical"] > 5)
    True

    Under-noising relative to the claimed budget is likewise detected.

    >>> weak = lambda X, rng: float(X.sum() + rng.laplace(0, 0.05))
    >>> bool(epsilon_dp(weak, D, Dp, n_samples=20000, seed=0)["epsilon_empirical"]
    ...      > epsilon_dp(lap, D, Dp, n_samples=20000, seed=0)["epsilon_empirical"])
    True
    """
    rng = np.random.default_rng(seed)
    a = np.asarray([np.ravel(mech(np.asarray(D, dtype=float), rng))[0]
                    for _ in range(int(n_samples))], dtype=float)
    b = np.asarray([np.ravel(mech(np.asarray(D_prime, dtype=float), rng))[0]
                    for _ in range(int(n_samples))], dtype=float)
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if lo == hi:
        return RichResult(
            title="Empirical epsilon",
            summary_lines=[("epsilon (empirical)", float("inf"))],
            warnings=["the mechanism is deterministic; it provides no privacy"],
            payload={"epsilon_empirical": float("inf"), "max_log_ratio": float("inf"),
                     "n_usable_bins": 0, "n_excluded_bins": 0,
                     "method": "epsilon_dp"},
        )
    edges = np.linspace(lo, hi, int(bins) + 1)
    ca, _ = np.histogram(a, bins=edges)
    cb, _ = np.histogram(b, bins=edges)
    # A bin with a handful of samples inflates the ratio through noise alone.
    floor = max(10, int(0.001 * n_samples))
    usable = (ca >= floor) & (cb >= floor)
    if not np.any(usable):
        ratio = float("inf")
    else:
        pa = ca[usable] / ca.sum()
        pb = cb[usable] / cb.sum()
        ratio = float(np.max(np.abs(np.log(pa / pb))))
    return RichResult(
        title="Empirical epsilon",
        summary_lines=[("epsilon (empirical)", ratio),
                       ("usable bins", int(usable.sum()))],
        warnings=["this is a LOWER bound from one dataset pair: it can "
                  "disprove a claimed guarantee but never establish one"],
        payload={
            "epsilon_empirical": ratio, "max_log_ratio": ratio,
            "n_usable_bins": int(usable.sum()),
            "n_excluded_bins": int((~usable).sum()),
            "n_samples": int(n_samples), "method": "epsilon_dp",
        },
    )


def cheatsheet():
    return "dpepsm: LOWER bound only -- can disprove a guarantee, never certify one"


# compact alias per ledger/NAMING.md
epsilondp = epsilon_dp
