"""Test information function and the conditional standard error it implies."""

from math import sqrt

from ._richresult import RichResult
from ._k05irt import item_params, info

__all__ = ["test_information"]


def test_information(theta, a, b, c=None, upper=None, D=1.0):
    r"""Fisher information of the whole test at each ability value.

    .. math:: I(\theta)=\sum_{i=1}^{n} I_i(\theta),
              \qquad I_i(\theta)=\frac{[P_i'(\theta)]^2}{P_i(\theta)Q_i(\theta)}

    Item information is *additive* -- that is the property that makes
    IRT test assembly possible at all, since each item's contribution
    can be evaluated without reference to the rest of the form. For a
    2PL item this reduces to :math:`D^2a_i^2P_iQ_i`, peaking at
    :math:`\theta=b_i`; guessing (:math:`c_i>0`) both lowers the peak
    and shifts it above :math:`b_i`, because a correct response
    carries less information when it may have been a guess.

    The conditional standard error of measurement follows as
    :math:`SEM(\theta)=1/\sqrt{I(\theta)}`, the asymptotic standard
    error of the maximum-likelihood ability estimate.

    Parameters
    ----------
    theta : float or array-like
        Ability value(s).
    a, b : array-like
        Item discriminations and difficulties, one per item.
    c : array-like, optional
        Lower asymptotes (guessing). Default 0 -> 2PL.
    upper : array-like, optional
        Upper asymptotes. Default 1.
    D : float
        Scaling constant.

    Returns
    -------
    RichResult
        Keys ``information`` (list, one per theta), ``sem``,
        ``item_information`` (n_theta x n_items), ``theta``, ``n_items``.

    References
    ----------
    Lord, F. M. (1980). *Applications of Item Response Theory to
    Practical Testing Problems*. Erlbaum, ch. 5.
    Birnbaum, A. (1968). In Lord & Novick, *Statistical Theories of
    Mental Test Scores*, ch. 17.
    """
    aa, bb, cc, uu, n = item_params(a, b, c, upper)
    try:
        th = [float(t) for t in theta]
        scalar = False
    except TypeError:
        th = [float(theta)]
        scalar = True
    per_item = [[info(t, aa[i], bb[i], cc[i], uu[i], D) for i in range(n)] for t in th]
    tot = [sum(row) for row in per_item]
    sem = [1.0 / sqrt(v) if v > 0 else float("inf") for v in tot]
    return RichResult(
        payload={
            "information": tot[0] if scalar else tot,
            "sem": sem[0] if scalar else sem,
            "item_information": per_item[0] if scalar else per_item,
            "theta": th[0] if scalar else th,
            "n_items": n,
            "D": float(D),
            "method": "Test information function, sum of item informations",
        }
    )


def cheatsheet():
    return "tinfo: test information I(theta) = sum_i I_i(theta), and SEM = 1/sqrt(I)"
