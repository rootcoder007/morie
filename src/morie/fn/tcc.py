"""Test characteristic curve: expected total score as a function of theta."""

from ._richresult import RichResult
from ._k05irt import item_params, prob

__all__ = ["test_characteristic_curve"]


def test_characteristic_curve(theta, a, b, c=None, upper=None, D=1.0):
    r"""Expected number-correct score at each ability value.

    .. math:: T(\theta)=\sum_{i=1}^{n} P_i(\theta)

    the sum of the item response functions. ``T`` is the IRT
    true-score, and it is what links the latent metric back to the
    observed one: it is strictly increasing in theta, so it can be
    inverted to map a raw score onto an ability estimate, which is the
    basis of true-score equating.

    Its floor is :math:`\sum_i c_i`, not zero -- with guessing, an
    examinee at :math:`\theta=-\infty` still expects to get
    :math:`\sum c_i` items right.

    Parameters
    ----------
    theta : float or array-like
        Ability value(s).
    a, b : array-like
        Item discriminations and difficulties, one per item. A scalar
        ``a`` is broadcast (the 1PL/Rasch case).
    c : array-like, optional
        Lower asymptotes (guessing). Default 0 -> 2PL.
    upper : array-like, optional
        Upper asymptotes. Default 1.
    D : float
        Scaling constant; 1.0 logistic, 1.702 for the normal-ogive
        approximation.

    Returns
    -------
    RichResult
        Keys ``tcc`` (list, one per theta), ``theta``, ``n_items``,
        ``floor`` (= sum of c), ``ceiling`` (= sum of upper).

    References
    ----------
    Lord, F. M. (1980). *Applications of Item Response Theory to
    Practical Testing Problems*. Erlbaum, ch. 4.
    Birnbaum, A. (1968). In Lord & Novick, *Statistical Theories of
    Mental Test Scores*, chs. 17-20.
    """
    aa, bb, cc, uu, n = item_params(a, b, c, upper)
    try:
        th = [float(t) for t in theta]
        scalar = False
    except TypeError:
        th = [float(theta)]
        scalar = True
    vals = [sum(prob(t, aa[i], bb[i], cc[i], uu[i], D) for i in range(n)) for t in th]
    return RichResult(
        payload={
            "tcc": vals[0] if scalar else vals,
            "theta": th[0] if scalar else th,
            "n_items": n,
            "floor": sum(cc),
            "ceiling": sum(uu),
            "D": float(D),
            "method": "Test characteristic curve, sum of item response functions",
        }
    )


def cheatsheet():
    return "tcc: test characteristic curve T(theta) = sum_i P_i(theta)"
