"""Quantile (pinball) loss (Koenker & Bassett 1978; Koenker 2005)."""

from ._richresult import RichResult

__all__ = ["qrF", "pinball_loss"]


def qrF(y_true, y_pred, theta=0.5):
    """
    Mean quantile (check / pinball) loss of predictions.

    The asymmetric absolute loss that defines regression quantiles
    (Koenker & Bassett 1978, Sec. 2): for the residual
    u = y - y_hat,

        rho_theta(u) = theta * u        if u >= 0,
                       (theta - 1) * u  if u < 0,

    i.e., under-predictions are weighted theta and over-predictions
    1 - theta; theta = 1/2 gives half the absolute error.  The
    empirical minimizer of the mean loss over constant predictions
    is the theta-th sample quantile.

    Sources
    -------
    Koenker, R. & Bassett, G. (1978). Regression quantiles.
    *Econometrica*, 46(1), 33-50, Sec. 2 (local copy
    fetched-wave3/Koenker-RegressionQuantiles-1978.pdf).
    Koenker, R. (2005). *Quantile Regression*. Cambridge University
    Press (delivered).

    Parameters
    ----------
    y_true, y_pred : sequences of float
        Observations and predictions.
    theta : float
        Quantile level in (0, 1).

    Returns
    -------
    RichResult
        Keys: estimate (mean loss), total, losses, theta, n.
    """
    yt = [float(v) for v in y_true]
    yp = [float(v) for v in y_pred]
    n = len(yt)
    if len(yp) != n or n == 0:
        raise ValueError("y_true and y_pred must be non-empty and paired")
    theta = float(theta)
    if not (0.0 < theta < 1.0):
        raise ValueError("theta must be in (0, 1)")
    losses = []
    for a, b in zip(yt, yp):
        u = a - b
        losses.append(theta * u if u >= 0 else (theta - 1.0) * u)
    tot = sum(losses)
    return RichResult(payload={
        "estimate": tot / n,
        "total": tot,
        "losses": losses,
        "theta": theta,
        "n": n,
        "method": "quantile/pinball loss (Koenker-Bassett 1978)",
    })


# long descriptive alias (stub-era name)
pinball_loss = qrF


def cheatsheet():
    return "qrF: rho_theta(u) = u(theta - 1[u<0]); minimized by theta-quantile"
