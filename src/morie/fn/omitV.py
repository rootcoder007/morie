# morie.fn -- function file (rootcoder007/morie)
"""Omitted variable bias, classical and partial-R2 form."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['ovbias', 'omitted_variable_bias']


def ovbias(delta=None, gamma=None, estimate=None, se=None, df=None, r2_yz=None, r2_dz=None):
    """Omitted variable bias, classical and partial-R2 form.

    The classical form needs the confounder's two regression coefficients, which by construction you do not have for an unobserved confounder. The partial-R2 form is the same bias re-expressed in quantities you can reason about without seeing the confounder at all, which is the whole contribution. Both are returned; supply ``delta`` and ``gamma`` for the first, or ``estimate``/``se``/``df`` with the two partial R-squareds for the second. The adjusted standard error is the paper's equation (12), including its df/(df-1) factor, which is easy to drop and changes the t-value.


    Formula: bias = delta gamma; equivalently |bias| = se(tau_res) sqrt(R2_YZ.DX R2_DZ.X / (1 - R2_DZ.X)) sqrt(df)

    Parameters
    ----------
    delta : float, optional
        Coefficient of the confounder in the treatment regression.
    gamma : float, optional
        Coefficient of the confounder in the outcome regression.
    estimate : float, optional
        The estimate obtained without the confounder.
    se : float, optional
        Its standard error.
    df : int, optional
        Residual degrees of freedom of that regression.
    r2_yz : float, optional
        Partial R2 of the confounder with the outcome given treatment and covariates.
    r2_dz : float, optional
        Partial R2 of the confounder with the treatment given covariates.

    Returns
    -------
    RichResult
        ``bias``, ``adjusted_estimate``, ``adjusted_se``, ``adjusted_t``, ``relative_bias``, ``bias_factor``.

    References
    ----------
    Cinelli and Hazlett (2020), Making Sense of Sensitivity: Extending
    Omitted Variable Bias, JRSS-B 82:39-67.  Verified against the
    author's copy of the paper: bias = delta gamma (Section 4.1),
    equation (12) for the adjusted standard error, equation (13) for the
    bias in partial-R2 form, equation (14) for the relative bias.
    """
    bias = float("nan")
    if delta is not None and gamma is not None:
        bias = float(delta) * float(gamma)
    adj_se = adj_t = rel = bf = float("nan")
    if r2_yz is not None and r2_dz is not None and se is not None and df is not None:
        ry, rd, s, d = float(r2_yz), float(r2_dz), float(se), float(df)
        if not 0 <= ry < 1 or not 0 <= rd < 1:
            raise ValueError("partial R2 values must be in [0, 1)")
        if d <= 1:
            raise ValueError("df must exceed 1")
        bias = s * math.sqrt(ry * rd / (1.0 - rd)) * math.sqrt(d)
        adj_se = s * math.sqrt((1.0 - ry) / (1.0 - rd) * d / (d - 1.0))
        bf = math.sqrt(ry) * math.sqrt(rd / (1.0 - rd))
        if estimate is not None:
            f_yd = abs(float(estimate) / s) / math.sqrt(d)
            rel = bf / f_yd if f_yd > 0 else float("inf")
    adj = float("nan")
    if estimate is not None and bias == bias:
        e = float(estimate)
        adj = e - math.copysign(bias, e) if bias >= 0 else e - bias
        if adj_se == adj_se and adj_se > 0:
            adj_t = adj / adj_se
    return RichResult(payload={
        "bias": bias, "adjusted_estimate": adj, "adjusted_se": adj_se,
        "adjusted_t": adj_t, "relative_bias": rel, "bias_factor": bf,
        "method": "Omitted variable bias (Cinelli-Hazlett)"})


omitted_variable_bias = ovbias


def cheatsheet():
    return "omitV: Omitted variable bias, classical and partial-R2 form."
