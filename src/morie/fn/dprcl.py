# morie.fn -- function file (rootcoder007/morie)
"""Calibrate a privacy budget to a target accuracy."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["dp_release_calibration"]


def dp_release_calibration(sensitivity=1.0, target_error=None, epsilon=None,
                           confidence=0.95, n=1):
    r"""Solve the privacy-accuracy trade in whichever direction is needed.

    Given a target error, return the :math:`\varepsilon` that achieves it;
    given an :math:`\varepsilon`, return the error it implies. For the Laplace
    mechanism with scale :math:`b = \Delta/(n\varepsilon)`, a central
    :math:`\gamma` interval has half-width

    .. math::
        w = b \ln\!\left(\frac{1}{1-\gamma}\right)
          \;\Longrightarrow\;
          \varepsilon = \frac{\Delta \ln(1/(1-\gamma))}{n\,w}.

    Working backwards from a tolerable error is the honest way to set a
    budget, and it is the direction practitioners rarely take. Chosen the
    other way -- picking :math:`\varepsilon = 1` because it is conventional --
    the resulting error is whatever it happens to be, which may be larger than
    the effect being measured.

    An :math:`\varepsilon` far above about 10 is worth naming as such: the
    likelihood ratio :math:`e^{10} \approx 22000` means the guarantee is
    close to vacuous, whatever the number on the page.

    Parameters
    ----------
    sensitivity : float
        Query sensitivity, positive.
    target_error : float, optional
        Desired half-width. Supply this or ``epsilon``.
    epsilon : float, optional
        Budget, if solving in the other direction.
    confidence : float
        Interval level, in (0, 1).
    n : int
        Sample size, for a mean-style query.

    Returns
    -------
    RichResult
        ``epsilon``, ``half_width``, ``noise_scale``, ``direction``.

    References
    ----------
    Dwork, C., & Roth, A. (2014). The algorithmic foundations of
        differential privacy. *FnT-TCS*, 9(3-4), 211-487.

    Examples
    --------
    Solving in either direction is self-consistent: calibrate to an error,
    then read the error back off the resulting budget.

    >>> r = dp_release_calibration(sensitivity=1.0, target_error=0.01, n=1000)
    >>> back = dp_release_calibration(sensitivity=1.0, epsilon=r["epsilon"], n=1000)
    >>> bool(abs(back["half_width"] - 0.01) < 1e-12)
    True

    A tighter error demands a larger budget, which is the trade made explicit.

    >>> a = dp_release_calibration(1.0, target_error=0.1, n=100)["epsilon"]
    >>> b = dp_release_calibration(1.0, target_error=0.01, n=100)["epsilon"]
    >>> bool(b > a)
    True

    A budget that has become effectively meaningless is named as such.

    >>> bool(dp_release_calibration(1.0, target_error=1e-4, n=10).warnings)
    True

    >>> dp_release_calibration(1.0)
    Traceback (most recent call last):
        ...
    ValueError: supply exactly one of target_error or epsilon
    """
    if (target_error is None) == (epsilon is None):
        raise ValueError("supply exactly one of target_error or epsilon")
    sensitivity = float(sensitivity)
    if sensitivity <= 0:
        raise ValueError("sensitivity must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    n = int(n)
    if n < 1:
        raise ValueError("n must be at least 1")
    z = float(np.log(1.0 / (1.0 - confidence)))

    if target_error is not None:
        w = float(target_error)
        if w <= 0:
            raise ValueError("target_error must be positive")
        eps = sensitivity * z / (n * w)
        direction = "error -> epsilon"
    else:
        eps = float(epsilon)
        if eps <= 0:
            raise ValueError("epsilon must be positive")
        w = sensitivity * z / (n * eps)
        direction = "epsilon -> error"
    b = sensitivity / (n * eps)
    return RichResult(
        title="Privacy budget calibration",
        summary_lines=[("direction", direction), ("epsilon", eps),
                       ("half-width", w), ("n", n)],
        warnings=(["epsilon exceeds 10, so the likelihood ratio is above 22000 "
                   "and the guarantee is close to vacuous"] if eps > 10 else []),
        payload={
            "epsilon": eps, "half_width": w, "noise_scale": b,
            "noise_sd": float(np.sqrt(2.0) * b), "direction": direction,
            "sensitivity": sensitivity, "confidence": float(confidence),
            "n": n, "method": "dp_release_calibration",
        },
    )


def cheatsheet():
    return "dprcl: set the budget FROM a tolerable error, not by convention; eps > 10 is near-vacuous"
