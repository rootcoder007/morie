# morie.fn -- function file (rootcoder007/morie)
"""Renyi differential privacy of the Gaussian mechanism -- Mironov (2017)."""

from __future__ import annotations

from ._richresult import RichResult

__all__ = ["rdpc", "renyi_dp"]


def renyi_dp(alpha, sigma, sensitivity=1.0):
    r"""RDP budget curve of the Gaussian mechanism.

    Mironov (2017), Proposition 7:

    .. math::
        D_\alpha\bigl(N(0, \sigma^2)\,\|\,N(\mu, \sigma^2)\bigr)
            = \alpha \mu^2 / (2 \sigma^2),

    and Corollary 3: for a query of sensitivity :math:`\Delta` the
    Gaussian mechanism with noise scale :math:`\sigma` satisfies
    :math:`(\alpha, \alpha \Delta^2 / (2\sigma^2))`-RDP for every
    :math:`\alpha > 1`.  The budget curve is a straight line in
    :math:`\alpha`; composition of n identical Gaussian mechanisms has
    the curve of a single one with scale :math:`\sigma/\sqrt{n}`
    (remark after Corollary 3), which is the test-suite anchor.

    Conversion to (epsilon, delta)-DP is a separate step (Proposition 3,
    shipped in ``dprnyi.renyi_dp_composition``) and is deliberately not
    duplicated here.

    Parameters
    ----------
    alpha : float
        Renyi order, must exceed 1.
    sigma : float
        Noise standard deviation, positive.
    sensitivity : float, default 1.0
        L2 sensitivity of the query, positive.

    Returns
    -------
    RichResult
        ``epsilon_rdp``, ``alpha``, ``sigma``, ``sensitivity``.

    References
    ----------
    Mironov, I. (2017). Renyi differential privacy. *IEEE CSF 2017*,
        263-275. arXiv:1702.07476. Proposition 7, Corollary 3, Table II.
        Local source: /run/media/rootcoder/WD_BLACK/library/pdf/fetched-wave3/mironov-2017-renyi-differential-privacy-arxiv1702.07476.pdf
    """
    alpha = float(alpha)
    sigma = float(sigma)
    sensitivity = float(sensitivity)
    if not alpha > 1.0:
        raise ValueError("alpha must exceed 1")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if sensitivity <= 0.0:
        raise ValueError("sensitivity must be positive")
    eps = alpha * sensitivity * sensitivity / (2.0 * sigma * sigma)
    return RichResult(payload={
        "epsilon_rdp": eps, "estimate": eps, "alpha": alpha,
        "sigma": sigma, "sensitivity": sensitivity,
        "method": "Gaussian-mechanism RDP (Mironov 2017, Corollary 3)"})


#: Primary name for the module.
rdpc = renyi_dp


def cheatsheet():
    return "rdpc: Gaussian-mechanism Renyi DP budget (Mironov 2017, Corollary 3)."

# public names resolved by fn/_lazy_map.json
renyidp = renyi_dp
