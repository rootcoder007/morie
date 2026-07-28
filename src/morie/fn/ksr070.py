# morie.fn -- function file (rootcoder007/morie)
"""Score operator for a nuisance path."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kosorok_score_operator"]


def kosorok_score_operator(log_p, eta_path, x, h, t=0.0, step=1e-5):
    r"""The score operator along a nuisance path (Kosorok Eq. 3.10,
    p. 46):

    .. math:: B_{\theta,\eta}h(x) - P_{\theta,\eta}B_{\theta,\eta}h
              = \frac{d}{dt}\log p_{\theta,\eta_t(\theta,\eta)}(x)
              \Big|_{t=0}.

    :math:`B` maps a DIRECTION :math:`h` in the nuisance space to a
    score function, and its range is the nuisance tangent space --
    the object the efficient score is orthogonal to. Centring by
    :math:`P B h` is not cosmetic: scores have mean zero by
    construction, and the uncentred derivative does not.

    Computed here by differentiating the log density along the
    supplied path, so a caller supplies the model and the direction
    rather than a formula, and the returned ``mean`` should be
    numerically zero after centring -- which the module checks.

    Parameters
    ----------
    log_p : callable
        ``log_p(eta, x)``, the log density.
    eta_path : callable
        ``eta_path(t, h)``, a submodel through the nuisance at
        ``t = 0``.
    x : array-like
        Observations.
    h : object
        The direction.
    t : float
        Point on the path to differentiate at.
    step : float
        Central-difference step.

    Returns
    -------
    RichResult
        keys: ``score``, ``mean``, ``centred``, ``mean_after_centring``,
        ``range_is``, ``n``, ``method``.
    References
    ----------
    Kosorok, Ch. 3, Eq. (3.10), p. 46.
    """
    xs = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    dt = float(step)
    if dt <= 0:
        raise ValueError(f"step must be positive, got {dt}.")
    up = np.asarray([float(log_p(eta_path(t + dt, h), v)) for v in xs])
    dn = np.asarray([float(log_p(eta_path(t - dt, h), v)) for v in xs])
    raw = (up - dn) / (2.0 * dt)
    mean = float(raw.mean())
    centred = raw - mean
    return RichResult(payload={
        "score": raw, "mean": mean, "centred": centred,
        "mean_after_centring": float(centred.mean()),
        "range_is": "the nuisance tangent space, which the efficient score is orthogonal to",
        "centring_note": "scores have mean zero by construction; the raw derivative does not",
        "n": int(xs.size),
        "method": "Score operator (Eq. 3.10); maps a nuisance DIRECTION to a score"})


def cheatsheet():
    return "ksr070: B maps directions to scores, and its RANGE is the nuisance tangent space"
