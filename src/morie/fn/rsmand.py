"""Andrich rating scale model for one item: category probabilities."""

from __future__ import annotations

from ._irtcore import cat_moments, rsm_probs, seq_
from ._richresult import RichResult

__all__ = ["rating_scale_andrich"]


def rating_scale_andrich(theta, b=0.0, tau=(0.0,)):
    r"""Category probabilities of one item under Andrich's rating scale model.

    For an item with categories :math:`h = 0, \ldots, m`, difficulty
    :math:`b` and a set of thresholds :math:`\tau_1, \ldots, \tau_m` that
    are shared across items,

    .. math::
        P(X = h \mid \theta)
          = \frac{\exp\{\sum_{j=1}^{h} (\theta - b - \tau_j)\}}
                 {\sum_{l=0}^{m} \exp\{\sum_{j=1}^{l} (\theta - b - \tau_j)\}},

    with the empty sum for :math:`h = 0` equal to zero. Writing
    :math:`T_h = \sum_{j\le h}\tau_j` the numerator exponent is
    :math:`h(\theta - b) - T_h`.

    This is Mair & Hatzinger's eq. (5) up to the sign convention on the item
    parameter: they write :math:`\exp[h(\theta_v + \beta_i) + \omega_h]`
    with an easiness :math:`\beta_i` and category parameters
    :math:`\omega_h`, which is this formula with
    :math:`\beta_i = -b` and :math:`\omega_h = -T_h`.

    The previous body was a placeholder: it averaged a leading ``y``
    argument and never referenced ``theta``, ``b`` or ``tau_j``. That
    spurious ``y`` is gone.

    Parameters
    ----------
    theta : float or array-like
        Ability values.
    b : float, default 0.0
        Item difficulty (location).
    tau : array-like, default (0.0,)
        The ``m`` category thresholds, giving ``m + 1`` categories. Note
        that only ``tau`` differences matter for the shape; adding a
        constant to every :math:`\tau_j` shifts the item location.

    Returns
    -------
    RichResult
        ``p`` (one list of ``m + 1`` probabilities per theta), ``eta``,
        ``expected`` (the expected category score per theta), ``info``
        (item information, which for this model is exactly the variance of
        the category score), ``theta``, ``b``, ``tau``, ``ncat``, ``n``,
        ``method``.

    Notes
    -----
    With ``m = 1`` the model collapses exactly to the dichotomous Rasch
    model, :math:`P(X=1) = \mathrm{expit}(\theta - b - \tau_1)`.

    References
    ----------
    Andrich, D. (1978). A rating formulation for ordered response
    categories. *Psychometrika*, 43(4), 561-573. doi:10.1007/BF02293814

    Mair, P. & Hatzinger, R. (2007). Extended Rasch modeling: the eRm
    package for the application of IRT models in R. *Journal of Statistical
    Software*, 20(9), eq. (5), p. 4.
    """
    th = [float(v) for v in seq_(theta)]
    n = len(th)
    if n == 0:
        raise ValueError("theta is empty.")
    tv = [float(v) for v in seq_(tau)]
    if len(tv) == 0:
        raise ValueError("tau is empty; a rating scale needs at least one threshold.")
    bb = float(b)
    scores = list(range(len(tv) + 1))

    p = []
    eta = []
    expected = []
    info = []
    for t in th:
        pr, et = rsm_probs(t, bb, tv)
        mu, v = cat_moments(pr, scores)
        p.append(pr)
        eta.append(et)
        expected.append(mu)
        info.append(v)

    return RichResult(
        payload={
            "p": p,
            "eta": eta,
            "expected": expected,
            "info": info,
            "theta": th,
            "b": bb,
            "tau": tv,
            "ncat": len(tv) + 1,
            "n": n,
            "method": "Andrich rating scale model, one item (Andrich 1978)",
        }
    )


def cheatsheet():
    return ("rsmand: Andrich RSM  P(X=h) proportional to "
            "exp(h(theta-b) - sum_{j<=h} tau_j)")
