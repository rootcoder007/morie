"""Gaussian-prior posterior for basis coefficients (ESL sec 8.3)."""

import math

from ._richresult import RichResult
from .nlsgn import _inverse

__all__ = ["esl_bayes_basis_posterior"]


def esl_bayes_basis_posterior(H, y, sigma2, tau, Sigma=None, Hnew=None):
    r"""Posterior of :math:`\beta` under :math:`\beta \sim N(0, \tau\Sigma)` and Gaussian noise.

    ESL eqs 8.25-8.28: with :math:`A = H^TH + (\sigma^2/\tau)\Sigma^{-1}`,
    :math:`E(\beta|Z) = A^{-1}H^Ty`, :math:`\mathrm{cov}(\beta|Z) = A^{-1}\sigma^2`,
    and for :math:`\mu(x) = h(x)^T\beta` the posterior mean
    :math:`h(x)^TA^{-1}H^Ty` and standard deviation
    :math:`[h(x)^TA^{-1}h(x)\sigma^2]^{1/2}`. As :math:`\tau\to\infty` this is least squares.

    Parameters
    ----------
    H : N x p nested sequence
    y : sequence of N floats
    sigma2 : float
        Noise variance, > 0.
    tau : float
        Prior scale, > 0.
    Sigma : p x p nested sequence, optional
        Prior correlation (default identity).
    Hnew : M x p nested sequence, optional

    Returns
    -------
    RichResult
        ``mean``, ``cov`` (p x p), ``mu_mean``, ``mu_sd``.

    References
    ----------
    Hastie, Tibshirani & Friedman (2009), sec. 8.3.
    """
    rows = [[float(v) for v in r] for r in H]
    yy = [float(v) for v in y]
    n, p = len(rows), len(rows[0])
    if len(yy) != n or sigma2 <= 0 or tau <= 0:
        raise ValueError("need matching H and y, sigma2 > 0 and tau > 0")
    S = (
        [[1.0 if a == b else 0.0 for b in range(p)] for a in range(p)]
        if Sigma is None
        else [[float(v) for v in r] for r in Sigma]
    )
    Si = _inverse(S)
    A = [[sum(r[a] * r[b] for r in rows) + sigma2 / tau * Si[a][b] for b in range(p)] for a in range(p)]
    Ai = _inverse(A)
    hty = [sum(r[a] * v for r, v in zip(rows, yy)) for a in range(p)]
    mean = [sum(Ai[a][b] * hty[b] for b in range(p)) for a in range(p)]
    Q = rows if Hnew is None else [[float(v) for v in r] for r in Hnew]
    return RichResult(
        title="Bayesian basis-coefficient posterior",
        summary_lines=[("p", p)],
        payload={
            "mean": mean,
            "cov": [[v * sigma2 for v in r] for r in Ai],
            "mu_mean": [sum(m * v for m, v in zip(mean, r)) for r in Q],
            "mu_sd": [math.sqrt(sigma2 * sum(r[a] * Ai[a][b] * r[b] for a in range(p) for b in range(p))) for r in Q],
        },
    )


def cheatsheet():
    return "eslbbp: E(beta|Z) = (H'H + s2/tau Sigma^-1)^-1 H'y, cov = (...)^-1 s2; ESL 8.27"
