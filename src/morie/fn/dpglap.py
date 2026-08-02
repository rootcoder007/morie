# morie.fn -- function file (rootcoder007/morie)
"""Laplace mechanism -- Dwork et al. (2006)."""

from __future__ import annotations

from . import _array_core as np

from ._dp import check_budget
from ._richresult import RichResult

__all__ = ["dp_laplace_mechanism"]


def dp_laplace_mechanism(y, sensitivity=1.0, epsilon=1.0, seed=None):
    r"""Release ``y`` under pure :math:`\varepsilon`-differential privacy.

    Adds :math:`\mathrm{Laplace}(0, \Delta_1/\varepsilon)` noise to each
    coordinate, where :math:`\Delta_1 = \max_{D \sim D'} \lVert f(D) -
    f(D')\rVert_1` is the L1 sensitivity.

    The sensitivity is the whole guarantee. It is a property of the *query*,
    not of the data in hand, and it must be an upper bound over all
    neighbouring datasets -- computing it from the observed data is itself a
    privacy leak and a common way to publish something that looks private and
    is not.

    Noise scale is :math:`b = \Delta_1/\varepsilon`, so the released value has
    standard deviation :math:`\sqrt2\, b` per coordinate. Halving
    :math:`\varepsilon` doubles the noise.

    Parameters
    ----------
    y : array-like or float
        The true query answer.
    sensitivity : float
        L1 sensitivity :math:`\Delta_1`, positive.
    epsilon : float
        Privacy budget, positive. Smaller is more private.
    seed : int, optional
        Seed. Leave as ``None`` for a real release -- a fixed seed makes the
        noise reproducible and therefore removable.

    Returns
    -------
    RichResult
        ``release``, ``noise_scale``, ``noise_sd``, ``epsilon``,
        ``sensitivity``.

    References
    ----------
    Dwork, C., McSherry, F., Nissim, K., & Smith, A. (2006). Calibrating noise
        to sensitivity in private data analysis. *TCC 2006*, 265-284.
    Dwork, C., & Roth, A. (2014). The algorithmic foundations of
        differential privacy. *Foundations and Trends in Theoretical
        Computer Science*, 9(3-4), 211-487.

    Examples
    --------
    Noise scale is sensitivity over epsilon, and the release is unbiased.

    >>> import numpy as np
    >>> r = dp_laplace_mechanism(100.0, sensitivity=1.0, epsilon=0.5, seed=0)
    >>> float(r["noise_scale"])
    2.0
    >>> draws = [dp_laplace_mechanism(100.0, 1.0, 0.5, seed=s)["release"]
    ...          for s in range(4000)]
    >>> bool(abs(float(np.mean(draws)) - 100.0) < 0.2)
    True

    Halving epsilon doubles the noise -- the privacy/accuracy trade in one line.

    >>> a = dp_laplace_mechanism(0.0, 1.0, 1.0, seed=1)["noise_scale"]
    >>> b = dp_laplace_mechanism(0.0, 1.0, 0.5, seed=1)["noise_scale"]
    >>> float(b / a)
    2.0

    >>> dp_laplace_mechanism(1.0, sensitivity=1.0, epsilon=0.0)
    Traceback (most recent call last):
        ...
    ValueError: epsilon must be finite and positive
    """
    epsilon, _ = check_budget(epsilon)
    sensitivity = float(sensitivity)
    if sensitivity <= 0:
        raise ValueError("sensitivity must be positive")
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    b = sensitivity / epsilon
    rel = y + rng.laplace(0.0, b, y.shape)
    return RichResult(
        title="Laplace mechanism",
        summary_lines=[("epsilon", epsilon), ("sensitivity", sensitivity),
                       ("noise scale", b)],
        payload={
            "release": rel if rel.ndim else float(rel),
            "noise_scale": b, "noise_sd": float(np.sqrt(2.0) * b),
            "epsilon": epsilon, "delta": 0.0, "sensitivity": sensitivity,
            "mechanism": "laplace", "method": "dp_laplace_mechanism",
        },
    )


def cheatsheet():
    return "dpglap: b = L1 sensitivity/epsilon; sensitivity is a property of the QUERY, never of the data"
