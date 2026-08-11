# SPDX-License-Identifier: AGPL-3.0-or-later
"""Variance partition coefficient for two-level models."""

import math

from ._richresult import RichResult

__all__ = ["vpc", "variance_partition_coefficient"]

_LOGISTIC_VAR = math.pi * math.pi / 3.0


def vpc(sigma2_u, sigma2_e=None, link="logistic"):
    """
    Variance partition coefficient (VPC) for a two-level model.

    For the continuous-response variance-components model
    (Goldstein, Browne and Rasbash 2002, eq. 1)

        VPC = sigma2_u / (sigma2_u + sigma2_e).

    For a binary response modelled with a logit link, the latent
    variable approach ("Method D", their sec. 3.5) treats the observed
    0/1 as a thresholded underlying continuous variable whose level-1
    residual follows the standard logistic distribution (their eqs. 5-6)
    with variance pi^2 / 3 = 3.29, so

        VPC = sigma2_u / (sigma2_u + pi^2 / 3),

    and analogously for the probit link the standard normal latent
    residual has variance 1.

    Parameters
    ----------
    sigma2_u : float
        Level-2 (between-cluster) variance.
    sigma2_e : float, optional
        Level-1 residual variance; required for ``link="identity"``,
        ignored otherwise.
    link : str
        "identity" (continuous response), "logistic" (latent level-1
        variance pi^2/3) or "probit" (latent level-1 variance 1).

    Returns
    -------
    result : RichResult
        Keys: estimate, sigma2_u, sigma2_1, link, method.

    References
    ----------
    Goldstein, H., Browne, W. and Rasbash, J. (2002), "Partitioning
    variation in multilevel models", Understanding Statistics 1(4),
    223-231; preprint sec. 2 eq. (1) and sec. 3.5 "A latent variable
    approach (Method D)", eqs. (5)-(6) [source:
    library/pdf/fetched-wave3/
    goldstein-browne-rasbash-2002-partitioning-variation.pdf].
    Printed anchor: for their voting example (beta0 = -0.256,
    sigma2_u0 = 0.142) Method D gives VPC = 0.041 (their sec. 3.5).
    """
    s2u = float(sigma2_u)
    if s2u < 0:
        raise ValueError("sigma2_u must be nonnegative")
    link = str(link).lower()
    if link in ("identity", "gaussian", "normal", "continuous"):
        if sigma2_e is None:
            raise ValueError("sigma2_e required for identity link")
        s21 = float(sigma2_e)
        if s21 < 0:
            raise ValueError("sigma2_e must be nonnegative")
    elif link in ("logistic", "logit"):
        s21 = _LOGISTIC_VAR
    elif link == "probit":
        s21 = 1.0
    else:
        raise ValueError("link must be identity, logistic or probit")
    est = s2u / (s2u + s21) if (s2u + s21) > 0 else float("nan")
    return RichResult(
        payload={
            "estimate": est,
            "sigma2_u": s2u,
            "sigma2_1": s21,
            "link": link,
            "method": "Variance partition coefficient (latent-variable Method D for binary links)",
        }
    )


def variance_partition_coefficient(sigma2_u, sigma2_e=None, link="logistic"):
    return vpc(sigma2_u, sigma2_e=sigma2_e, link=link)


def cheatsheet():
    return "vpc: variance partition coefficient (Goldstein-Browne-Rasbash 2002, Method D)"
