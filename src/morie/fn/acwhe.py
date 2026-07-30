# morie.fn -- function file (rootcoder007/morie)
"""Privacy-accuracy trade-off for a private mean."""

from __future__ import annotations

import numpy as np

from ._dp import check_budget
from ._richresult import RichResult

__all__ = ["private_accuracy_tradeoff"]


def private_accuracy_tradeoff(sensitivity=1.0, epsilon=1.0, n=100, confidence=0.95):
    r"""Expected error added by the Laplace mechanism, and the ``n`` needed to hide it.

    For a mean of ``n`` records with per-record sensitivity :math:`\Delta`, the
    Laplace noise has scale :math:`b = \Delta/(n\varepsilon)`, giving

    .. math::
        \mathrm{sd} = \sqrt2\, b, \qquad
        \text{half-width} = b \ln\!\left(\frac{1}{1-\gamma}\right)

    for a central :math:`\gamma` interval on the Laplace distribution.

    The number worth reporting is ``noise_to_signal_n``: the sample size at
    which privacy noise falls below sampling error. Below it the release is
    dominated by the mechanism and the statistic is not really measuring
    anything; above it privacy is close to free. Quoting :math:`\varepsilon`
    without that comparison is how privacy budgets get chosen by vibes.

    Parameters
    ----------
    sensitivity : float
        Per-record sensitivity :math:`\Delta`, positive.
    epsilon : float
        Privacy budget, positive.
    n : int
        Sample size, at least 1.
    confidence : float
        Central interval level, in (0, 1).

    Returns
    -------
    RichResult
        ``noise_scale``, ``noise_sd``, ``half_width``, ``relative_error``,
        ``noise_to_signal_n``.

    References
    ----------
    Dwork, C., & Roth, A. (2014). The algorithmic foundations of
        differential privacy. *FnT-TCS*, 9(3-4), 211-407.

    Examples
    --------
    Error falls linearly in n and in epsilon.

    >>> a = private_accuracy_tradeoff(1.0, 1.0, n=100)["noise_sd"]
    >>> b = private_accuracy_tradeoff(1.0, 1.0, n=1000)["noise_sd"]
    >>> bool(abs(a / b - 10.0) < 1e-9)
    True
    >>> c = private_accuracy_tradeoff(1.0, 0.1, n=100)["noise_sd"]
    >>> bool(abs(c / a - 10.0) < 1e-9)
    True

    The crossover sample size scales like 1/epsilon^2.

    >>> lo = private_accuracy_tradeoff(1.0, 0.1, n=100)["noise_to_signal_n"]
    >>> hi = private_accuracy_tradeoff(1.0, 1.0, n=100)["noise_to_signal_n"]
    >>> bool(lo > hi)
    True

    >>> private_accuracy_tradeoff(1.0, 1.0, n=0)
    Traceback (most recent call last):
        ...
    ValueError: n must be at least 1
    """
    epsilon, _ = check_budget(epsilon)
    sensitivity = float(sensitivity)
    if sensitivity <= 0:
        raise ValueError("sensitivity must be positive")
    n = int(n)
    if n < 1:
        raise ValueError("n must be at least 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    b = sensitivity / (n * epsilon)
    sd = float(np.sqrt(2.0) * b)
    hw = float(b * np.log(1.0 / (1.0 - confidence)))
    # Privacy noise sd sqrt(2)*D/(n*eps) equals sampling error D/sqrt(n) when
    # n = 2/eps^2 -- the point past which privacy is near-free.
    cross = float(2.0 / epsilon**2)
    return RichResult(
        title="Privacy-accuracy trade-off",
        summary_lines=[("epsilon", epsilon), ("n", n), ("noise sd", sd),
                       ("crossover n", cross)],
        payload={
            "noise_scale": float(b), "noise_sd": sd, "half_width": hw,
            "relative_error": float(sd / max(abs(sensitivity), 1e-300)),
            "noise_to_signal_n": cross,
            "epsilon": epsilon, "n": n, "sensitivity": sensitivity,
            "confidence": float(confidence),
            "method": "private_accuracy_tradeoff",
        },
    )


def cheatsheet():
    return "acwhe: privacy noise beats sampling error once n > 2/eps^2 -- report that, not just epsilon"
