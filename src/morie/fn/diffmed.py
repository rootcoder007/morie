# morie.fn -- function file (rootcoder007/morie)
"""Difference-in-coefficients mediation estimator."""

import numpy as np

from ._richresult import RichResult

__all__ = ["difference_in_coefficients"]


def difference_in_coefficients(c, c_prime, a=None, b=None, se_c=None,
                               se_c_prime=None, alpha=0.05):
    r"""Indirect effect as :math:`c - c'`.

    Fit the total effect and then the direct effect controlling for the
    mediator,

    .. math::
       Y = i_1 + cX + e_1, \qquad Y = i_2 + c'X + bM + e_2,

    and take :math:`c - c'`.

    For a CONTINUOUS outcome fitted by ordinary least squares this is
    an algebraic identity: :math:`c - c' = ab` exactly, in every
    sample, not merely in expectation. ``matches_product`` checks it
    when ``a`` and ``b`` are supplied, and a non-zero residual means
    the two models were not fitted on the same rows -- the usual cause
    being listwise deletion dropping different cases from each.

    The identity FAILS for logistic or probit outcomes, and not because
    of sampling error. Those coefficients are identified only up to
    scale, and conditioning on :math:`M` changes the residual variance
    and hence the scale, so :math:`c` and :math:`c'` are not on the
    same footing and their difference is not an effect. This is
    non-collapsibility, and it is why difference-in-coefficients was
    abandoned for binary outcomes in favour of the counterfactual
    definitions.

    Parameters
    ----------
    c, c_prime : float
        Total and direct effect coefficients.
    a, b : float, optional
        Enables the identity check.
    se_c, se_c_prime : float, optional
    alpha : float

    Returns
    -------
    RichResult
        ``indirect``, ``proportion_mediated``, ``matches_product``,
        ``identity_residual``, ``ci``.

    References
    ----------
    Judd and Kenny (1981), *Evaluation Review* 5:602-619.
    MacKinnon (2008), *Introduction to Statistical Mediation
    Analysis*, chapter 3, on the equivalence and where it breaks.

    Examples
    --------
    >>> float(difference_in_coefficients(0.7, 0.5)["indirect"])
    0.19999999999999996
    """
    cv, cp = float(c), float(c_prime)
    ind = cv - cp
    prod = None if (a is None or b is None) else float(a) * float(b)
    resid = None if prod is None else float(abs(ind - prod))
    ci = None
    if se_c is not None and se_c_prime is not None:
        # c and c' come from the same data, so their difference has a
        # covariance term the naive sum ignores; without it this is only
        # an upper bound on the width
        s = float(np.sqrt(float(se_c) ** 2 + float(se_c_prime) ** 2))
        z = 1.959963984540054
        ci = (ind - z * s, ind + z * s)
    return RichResult(
        payload={
            "estimate": ind,
            "indirect": ind,
            "total": cv,
            "direct": cp,
            "proportion_mediated": float(ind / cv) if cv != 0 else np.nan,
            "proportion_note": (
                "the proportion mediated is unstable when the total effect "
                "is near zero, and can exceed 1 or go negative under "
                "inconsistent mediation, where direct and indirect paths "
                "have opposite signs"
            ),
            "product": prod,
            "matches_product": (None if resid is None else bool(resid < 1e-8)),
            "identity_residual": resid,
            "identity_note": (
                "for a continuous OLS outcome c - c' = ab exactly, in every "
                "sample; a non-zero residual means the two models were not "
                "fitted on the same rows"
            ),
            "ci": ci,
            "ci_note": (
                None if ci is None else
                "c and c' are estimated on the same data and are correlated; "
                "ignoring that covariance makes this an upper bound on the "
                "width"
            ),
            "binary_outcome_warning": (
                "for logistic or probit outcomes the identity fails "
                "systematically: those coefficients are identified only up "
                "to scale, and conditioning on M changes the residual "
                "variance and so the scale. Use a counterfactual definition "
                "instead"
            ),
            "method": "Difference-in-coefficients indirect effect",
        }
    )


def cheatsheet():
    return (
        "diffmed: c - c' with the exact OLS identity against ab and the "
        "non-collapsibility warning for binary outcomes"
    )
