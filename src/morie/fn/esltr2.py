"""Test-set mean squared error and R^2 relative to the constant model (ESL eqs 9.23-9.24)."""

from ._richresult import RichResult

__all__ = ["esl_test_r2"]


def esl_test_r2(mu, fitted, ybar):
    r"""MSE of the fit and of the constant model against the true mean, and their relative reduction.

    ESL eq 9.23: :math:`MSE_0 = \mathrm{ave}_{x\in Test}(\bar y - \mu(x))^2`,
    :math:`MSE = \mathrm{ave}_{x\in Test}(\hat f(x) - \mu(x))^2`; eq 9.24:
    :math:`R^2 = (MSE_0 - MSE)/MSE_0`, the proportion of the constant model's
    error removed (it can be negative for a fit worse than the mean).

    Parameters
    ----------
    mu : sequence of floats
        True mean mu(x) at the test points.
    fitted : sequence of floats
        Model predictions at the test points.
    ybar : float
        Training mean (the constant model).

    Returns
    -------
    RichResult
        ``mse``, ``mse0``, ``r2``.

    References
    ----------
    Hastie, Tibshirani & Friedman (2009), sec. 9.4.2.
    """
    m = [float(v) for v in mu]
    f = [float(v) for v in fitted]
    if len(m) != len(f) or not m:
        raise ValueError("mu and fitted must be non-empty and of equal length")
    mse = sum((a - b) ** 2 for a, b in zip(f, m)) / len(m)
    mse0 = sum((float(ybar) - b) ** 2 for b in m) / len(m)
    if mse0 == 0:
        raise ValueError("the constant model is exact; R^2 is undefined")
    return RichResult(
        title="Test-set R^2",
        summary_lines=[("r2", (mse0 - mse) / mse0)],
        payload={"mse": mse, "mse0": mse0, "r2": (mse0 - mse) / mse0},
    )


def cheatsheet():
    return "esltr2: R2 = (MSE0 - MSE) / MSE0 against the true mean, ESL 9.24"
