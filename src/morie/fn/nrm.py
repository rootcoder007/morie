"""Bock nominal response model for one item: category probabilities."""

from __future__ import annotations

from ._irtcore import cat_moments, nrm_probs, seq_
from ._richresult import RichResult

__all__ = ["nominal_response_bock"]


def nominal_response_bock(theta, a_k=(0.0, 1.0), c_k=(0.0, 0.0)):
    r"""Category probabilities of one item under Bock's nominal response model.

    .. math::
        P(X = r \mid \theta)
          = \frac{\exp(a_r \theta + c_r)}{\sum_{s} \exp(a_s \theta + c_s)}

    with a slope :math:`a_r` and an intercept :math:`c_r` for every
    response category, ordered or not.

    The previous body was a placeholder: it averaged a leading ``y``
    argument and never referenced ``theta``, ``a_k`` or ``c_k``. That
    spurious ``y`` is gone.

    Parameters
    ----------
    theta : float or array-like
        Ability values.
    a_k : array-like, default (0.0, 1.0)
        Category slopes, one per category.
    c_k : array-like, default (0.0, 0.0)
        Category intercepts, same length as ``a_k``.

    Returns
    -------
    RichResult
        ``p`` (one list of probabilities per theta), ``eta``, ``expected``
        (the mean slope :math:`\sum_r p_r a_r`, which is the derivative of
        the log normaliser), ``info`` (the item information, exactly the
        variance of :math:`a_R`), ``theta``, ``a_k``, ``c_k``, ``ncat``,
        ``n``, ``method``.

    Notes
    -----
    The model is invariant to adding a constant to every :math:`a_r` or to
    every :math:`c_r`, so it is not identified without a constraint; the
    usual one is :math:`\sum_r a_r = \sum_r c_r = 0`. None is imposed here
    -- the probabilities are unaffected either way.

    With two categories and :math:`a = (0, a_1)`, :math:`c = (0, c_1)` the
    model is exactly the 2PL, :math:`P(X=1) = \mathrm{expit}(a_1\theta + c_1)`.
    Setting :math:`a_r = r` gives the partial credit model.

    References
    ----------
    Bock, R. D. (1972). Estimating item parameters and latent ability when
    responses are scored in two or more nominal categories. *Psychometrika*,
    37(1), 29-51. doi:10.1007/BF02291411

    Tutz, G. (2020). A taxonomy of polytomous item response models.
    arXiv:2010.01382, eq. (14), p. 16, which prints Bock's model as
    :math:`\exp(\alpha_{ir}\theta_p - \beta_{ir})` over the same sum;
    :math:`c_r = -\beta_{ir}` here.
    """
    th = [float(v) for v in seq_(theta)]
    n = len(th)
    if n == 0:
        raise ValueError("theta is empty.")
    av = [float(v) for v in seq_(a_k)]
    cv = [float(v) for v in seq_(c_k)]
    if len(av) < 2:
        raise ValueError("a_k needs at least two categories.")
    if len(cv) != len(av):
        raise ValueError("c_k has length %d; expected %d to match a_k"
                         % (len(cv), len(av)))

    p = []
    eta = []
    expected = []
    info = []
    for t in th:
        pr, et = nrm_probs(t, av, cv)
        mu, v = cat_moments(pr, av)
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
            "a_k": av,
            "c_k": cv,
            "ncat": len(av),
            "n": n,
            "method": "Bock nominal response model, one item (Bock 1972)",
        }
    )


def cheatsheet():
    return "nrm: Bock NRM  P(X=r) = exp(a_r theta + c_r) / sum_s exp(a_s theta + c_s)"
