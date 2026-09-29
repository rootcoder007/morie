"""Per-legislator IRT variance from posterior."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult


def irt_variance_legislator(chain_theta) -> DescriptiveResult:
    r"""Posterior variance of each legislator's ideal point from MCMC draws.

    ``chain_theta`` is ``S x J`` (draws by legislators); the Monte Carlo
    estimate of ``Var(theta_j | data)`` is the sample variance of the ``S``
    draws with the ``S - 1`` divisor (Clinton, Jackman and Rivers 2004;
    Gelman et al. 2013, sec. 10.5). ``value`` is the vector of variances;
    ``extra`` also has their mean and the posterior SDs.

    References
    ----------
    Clinton, J., Jackman, S. and Rivers, D. (2004). The statistical analysis
    of roll call data. *American Political Science Review* 98, 355-370.

    Examples
    --------
    >>> [round(v, 12) for v in irt_variance_legislator([[0.1, 1.0], [0.3, 1.4], [0.2, 0.6]]).value]
    [0.01, 0.16]
    """
    rows = chain_theta.tolist() if hasattr(chain_theta, "tolist") else list(chain_theta)
    chain = [[float(v) for v in (r if hasattr(r, "__len__") else [r])] for r in rows]
    S, J = len(chain), len(chain[0])
    if S < 2:
        raise ValueError("at least two draws are needed")
    var = []
    for j in range(J):
        col = [r[j] for r in chain]
        m = math.fsum(col) / S
        var.append(math.fsum((v - m) ** 2 for v in col) / (S - 1))
    return DescriptiveResult(
        name="irt_variance_legislator",
        value=var,
        extra={
            "variances": var,
            "sd": [math.sqrt(v) for v in var],
            "mean_variance": math.fsum(var) / J,
            "n_legislators": J,
            "n_samples": S,
        },
    )


irtva = irt_variance_legislator


def cheatsheet() -> str:
    return "irt_variance_legislator(chain) -> posterior variance of each legislator's theta (S - 1 divisor)."
