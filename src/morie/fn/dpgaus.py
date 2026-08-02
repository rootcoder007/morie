# morie.fn -- function file (rootcoder007/morie)
"""Gaussian mechanism -- Dwork & Roth (2014) Appendix A."""

from __future__ import annotations

from . import _array_core as np

from ._dp import check_budget, gaussian_sigma
from ._richresult import RichResult

__all__ = ["dp_gaussian_mechanism"]


def dp_gaussian_mechanism(y, sensitivity=1.0, epsilon=1.0, delta=1e-5, seed=None):
    r"""Release ``y`` under :math:`(\varepsilon, \delta)`-differential privacy.

    Adds :math:`\mathcal{N}(0, \sigma^2)` noise with

    .. math::
        \sigma = \frac{\Delta_2 \sqrt{2\ln(1.25/\delta)}}{\varepsilon},

    calibrated to the **L2** sensitivity, not the L1 one the Laplace mechanism
    uses. For a :math:`d`-dimensional query whose coordinates each move by 1,
    :math:`\Delta_1 = d` while :math:`\Delta_2 = \sqrt d`, which is exactly
    why the Gaussian mechanism wins in high dimension despite needing
    :math:`\delta > 0`.

    The classical bound above is only valid for :math:`\varepsilon \le 1`.
    Above that it is not a proof of anything, so this warns rather than
    returning under-noised output.

    :math:`\delta` should be well below :math:`1/n`: a mechanism may release
    the whole database with probability :math:`\delta` and still satisfy the
    definition.

    Parameters
    ----------
    y : array-like or float
        True query answer.
    sensitivity : float
        L2 sensitivity :math:`\Delta_2`, positive.
    epsilon : float
        Privacy budget, positive.
    delta : float
        Failure probability, in (0, 1).
    seed : int, optional
        Seed; leave ``None`` for a real release.

    Returns
    -------
    RichResult
        ``release``, ``sigma``, ``epsilon``, ``delta``, ``sensitivity``.

    References
    ----------
    Dwork, C., & Roth, A. (2014). The algorithmic foundations of
        differential privacy. *Foundations and Trends in Theoretical
        Computer Science*, 9(3-4), 211-487.

    Examples
    --------
    >>> import numpy as np
    >>> r = dp_gaussian_mechanism(10.0, sensitivity=1.0, epsilon=1.0,
    ...                           delta=1e-5, seed=0)
    >>> bool(abs(r["sigma"] - np.sqrt(2 * np.log(1.25e5))) < 1e-9)
    True

    Unbiased across releases.

    >>> d = [dp_gaussian_mechanism(10.0, 1.0, 1.0, 1e-5, seed=s)["release"]
    ...      for s in range(4000)]
    >>> bool(abs(float(np.mean(d)) - 10.0) < 0.2)
    True

    Pure epsilon-DP is refused rather than silently granted.

    >>> dp_gaussian_mechanism(1.0, 1.0, 1.0, delta=0.0)
    Traceback (most recent call last):
        ...
    ValueError: the Gaussian mechanism needs delta > 0; use the Laplace mechanism for pure epsilon-DP

    Beyond epsilon = 1 the classical bound does not apply, and that is said.

    >>> r = dp_gaussian_mechanism(1.0, 1.0, epsilon=4.0, delta=1e-5, seed=0)
    >>> bool(r.warnings)
    True
    """
    epsilon, delta = check_budget(epsilon, delta)
    sensitivity = float(sensitivity)
    if sensitivity <= 0:
        raise ValueError("sensitivity must be positive")
    sigma = gaussian_sigma(sensitivity, epsilon, delta)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    rel = y + rng.normal(0.0, sigma, y.shape)
    warn = []
    if epsilon > 1.0:
        warn.append(
            f"the classical Gaussian bound requires epsilon <= 1 but epsilon={epsilon:g}; "
            "sigma here is not a proof of (epsilon, delta)-DP -- use the analytic "
            "Gaussian mechanism or split the budget"
        )
    return RichResult(
        title="Gaussian mechanism",
        summary_lines=[("epsilon", epsilon), ("delta", delta), ("sigma", sigma)],
        warnings=warn,
        payload={
            "release": rel if rel.ndim else float(rel),
            "sigma": sigma, "noise_sd": sigma,
            "epsilon": epsilon, "delta": delta, "sensitivity": sensitivity,
            "mechanism": "gaussian", "method": "dp_gaussian_mechanism",
        },
    )


def cheatsheet():
    return "dpgaus: calibrated to L2 (not L1) sensitivity; classical bound only valid for epsilon <= 1"
