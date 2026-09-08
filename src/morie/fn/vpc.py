"""Variance partition coefficient for multilevel binary models (Goldstein et al. 2002)."""

import math

from ._richresult import RichResult

__all__ = ["vpc", "variance_partition_coefficient"]


def vpc(sigma2_u, link="logit"):
    """
    Variance partition coefficient by the latent-variable method.

    Goldstein, Browne & Rasbash (2002), Method D: view the binary
    response as a thresholded latent continuous variable; for the
    logit link the level-1 residual follows the standard logistic
    distribution whose variance is pi^2 / 3 = 3.29 (their Eq. 5 and
    the surrounding text), so

        VPC = sigma^2_u / (sigma^2_u + pi^2 / 3).

    For the probit link the latent residual is standard normal with
    variance 1, giving VPC = sigma^2_u / (sigma^2_u + 1).  Their
    Table 1 (Method D column) reports level-2 variance 0.142 with
    level-1 variance 3.290 and VPC 0.043 for the worked example.

    Sources
    -------
    Goldstein, H., Browne, W. & Rasbash, J. (2002). Partitioning
    variation in multilevel models. *Understanding Statistics*,
    1(4), 223-231, Sec. 3.4 (Method D) and Table 1 (local copy
    fetched-wave3/goldstein-browne-rasbash-2002-vpc.pdf).

    Parameters
    ----------
    sigma2_u : float
        Level-2 (cluster) variance on the latent scale.
    link : str
        "logit" (default; level-1 variance pi^2/3) or "probit"
        (level-1 variance 1).

    Returns
    -------
    RichResult
        Keys: estimate (the VPC), sigma2_u, sigma2_e (level-1
        latent variance), link.
    """
    s2u = float(sigma2_u)
    if s2u < 0:
        raise ValueError("sigma2_u must be non-negative")
    lk = str(link).lower()
    if lk == "logit":
        s2e = math.pi ** 2 / 3.0
    elif lk == "probit":
        s2e = 1.0
    else:
        raise ValueError("link must be 'logit' or 'probit'")
    return RichResult(payload={
        "estimate": s2u / (s2u + s2e),
        "sigma2_u": s2u,
        "sigma2_e": s2e,
        "link": lk,
        "method": "latent-variable VPC (Goldstein et al. 2002, Method D)",
    })


# long descriptive alias (stub-era name)
variance_partition_coefficient = vpc


def cheatsheet():
    return "vpc: sigma2_u / (sigma2_u + pi^2/3) [logit] or (+1) [probit]"
