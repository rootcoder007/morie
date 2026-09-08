# morie.fn -- function file (rootcoder007/morie)
"""Composite likelihood estimation of semivariogram parameters."""

from . import _array_core as np

from ._richresult import RichResult
from ._schaben import MODELS, composite_likelihood_fit

__all__ = ["schabenberger_composite_likelihood"]


def schabenberger_composite_likelihood(coords, z, variogram_model="exponential"):
    r"""Composite-likelihood semivariogram fit, Schabenberger eq (4.44).

    The composite likelihood score is

    .. math::
       CS(\theta) = 2\sum_{i<j}
         \frac{\partial\gamma(h_{ij},\theta)}{\partial\theta}
         \frac{1}{8\gamma(h_{ij},\theta)^2}
         \left\{T^{(3)}_{ij} - 2\gamma(h_{ij},\theta)\right\},

    with :math:`T^{(3)}_{ij} = \{Z(s_i)-Z(s_j)\}^2`. Comparing this to
    the generalised estimating equation (4.43), the two differ ONLY by
    the factor :math:`1/(8\gamma^2)`, and that factor is not a tuning
    choice: under the Gaussian assumption
    :math:`T^{(3)}_{ij}/2\gamma(h_{ij},\theta)\sim\chi^2_1`, so
    :math:`\mathrm{Var}[T^{(3)}_{ij}] = 8\gamma(h_{ij},\theta)^2`
    exactly. The composite likelihood is the variance-weighted GEE,
    and the GEE with an identity working structure is the same
    estimator with that dispersion ignored.

    The practical consequence is the reason to prefer this route: the
    fit is to the semivariogram CLOUD, pair by pair. No lag classes
    are formed, so no binning choice, tolerance or cutoff can move the
    answer -- the arbitrariness that section 4.4.1 spends several pages
    warning about simply does not arise.

    What is given up is that the composite likelihood is not a real
    likelihood. The pairs are not independent, so the sum of component
    scores is an unbiased estimating function but not a score, and its
    value cannot be used for likelihood-ratio comparison.

    Parameters
    ----------
    coords : array-like, shape (n, d)
    z : array-like, shape (n,)
    variogram_model : {'exponential', 'spherical', 'gaussian', 'linear'}

    Returns
    -------
    RichResult
        ``parameters`` (nugget, psill, range), ``nugget``, ``psill``,
        ``range``, ``sill``, ``n_pairs``, ``objective``.

    References
    ----------
    Schabenberger and Gotway (2005), section 4.5.3, equations (4.43)
    and (4.44), pp. 169-172. Lindsay (1988). Curriero and Lele (1999).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> co = rng.uniform(0, 10, size=(60, 2))
    >>> z = rng.normal(size=60)
    >>> out = schabenberger_composite_likelihood(co, z)
    >>> bool(out["sill"] > 0 and out["range"] > 0)
    True
    """
    model = variogram_model
    if model not in MODELS:
        raise ValueError("model must be one of %s, got %r." % (MODELS, model))
    fit = composite_likelihood_fit(coords, z, model)
    return RichResult(
        payload={
            "estimate": np.array([fit["nugget"], fit["psill"], fit["range"]]),
            "parameters": {"nugget": fit["nugget"], "psill": fit["psill"],
                           "range": fit["range"]},
            "nugget": fit["nugget"],
            "psill": fit["psill"],
            "range": fit["range"],
            "sill": fit["sill"],
            "model": model,
            "n_pairs": fit["n_pairs"],
            "objective": fit["objective"],
            "iterations": fit["iterations"],
            "converged": fit["converged"],
            "convergence_note": fit["diverged_note"],
            "binning_note": (
                "fitted to the semivariogram cloud, so no lag classes, "
                "tolerance or cutoff enter the estimate"
            ),
            "likelihood_note": (
                "a composite likelihood is an unbiased estimating function, "
                "not a likelihood; its value does not support "
                "likelihood-ratio comparison"
            ),
            "n": int(np.asarray(z).size),
            "method": "Composite-likelihood semivariogram estimation",
        }
    )


def cheatsheet():
    return (
        "spclk: semivariogram by composite likelihood (4.44) -- the "
        "variance-weighted GEE, fitted to the cloud so binning cannot "
        "move the answer"
    )
