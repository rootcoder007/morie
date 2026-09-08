# morie.fn -- function file (rootcoder007/morie)
"""Theorem 4.3: bias and variance of boundary-free MRL estimators."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["fauzi_theorem_4_3", "fauzi_thm4_3_mrl_bias_var"]


def fauzi_theorem_4_3(t, S_X, S_bar_X, m_X, b1, b2=None, b3=None, n=100,
                      h=0.1, g_prime=1.0, f_X=1.0, mu2=1.0, VW=None):
    r"""Theorem 4.3 (Fauzi Eqs. 4.25-4.28): the biases and variances
    of the boundary-free mean-residual-life estimators.

    .. math:: \mathrm{Bias}[\tilde m_{X,1}(t)]
              = \frac{h^2}{2S_X(t)}\big[b_2(t) + m_X(t)b_1(t)\big]
                \int y^2K(y)dy + o(h^2),

    .. math:: \mathrm{Bias}[\tilde m_{X,2}(t)]
              = \frac{h^2}{2S_X(t)}\big[b_3(t) + m_X(t)b_1(t)\big]
                \int y^2K(y)dy + o(h^2),

    .. math:: \mathrm{Var}[\tilde m_{X,i}(t)]
              = \frac1n\frac{b_4(t)}{S_X^2(t)}
              - \frac hn\frac{b_5(t)}{S_X^2(t)}\int V(y)W(y)dy
              + o\!\left(\frac hn\right),

    with :math:`b_4(t) = 2\bar{\mathbb S}_X(t) - S_X(t)m_X^2(t)` and
    :math:`b_5(t) = g'(g^{-1}(t))f_X(t)m_X^2(t)`.

    Two structural facts are visible in the formulas rather than
    stated separately. The two estimators differ ONLY in whether
    :math:`b_2` or :math:`b_3` appears -- everything else is shared,
    which is why Remark 4.3 says their statistical behaviour matches.
    And the variance's leading term is :math:`O(1/n)` with the
    bandwidth entering only at order :math:`h/n`, so the variance is
    far less sensitive to the bandwidth than the bias is.

    Parameters
    ----------
    t : array-like
        Evaluation points.
    S_X, S_bar_X, m_X : array-like
        Survival function, its integral, and the MRL at ``t``.
    b1 : array-like
        The :math:`b_1` coefficient.
    b2, b3 : array-like, optional
        Coefficients for the first and second estimator.
    n : int
        Sample size.
    h : float
        Bandwidth.
    g_prime, f_X : array-like
        Transformation derivative and density at ``t``.
    mu2 : float
        :math:`\int y^2K(y)dy`.
    VW : float, optional
        :math:`\int V(y)W(y)dy`; the Gaussian value otherwise.

    Returns
    -------
    RichResult
        keys: ``t``, ``bias_1``, ``bias_2``, ``variance``, ``b4``,
        ``b5``, ``differ_only_in``, ``variance_leading_order``,
        ``n``, ``h``, ``method``.
    References
    ----------
    Fauzi and Maesono (2023), Theorem 4.3, Eqs. (4.25)-(4.28) and
    Remark 4.3. Transcribed from the PDF.
    """
    tv = np.atleast_1d(np.asarray(t, dtype=float)).ravel()
    S = np.atleast_1d(np.asarray(S_X, dtype=float)).ravel()
    Sb = np.atleast_1d(np.asarray(S_bar_X, dtype=float)).ravel()
    m = np.atleast_1d(np.asarray(m_X, dtype=float)).ravel()
    c1 = np.atleast_1d(np.asarray(b1, dtype=float)).ravel()
    for nm, arr in (("S_X", S), ("S_bar_X", Sb), ("m_X", m), ("b1", c1)):
        if arr.size != tv.size:
            raise ValueError(f"{nm} has {arr.size} entries for {tv.size}.")
    if np.any(S <= 0):
        raise ValueError("S_X must be strictly positive to divide by it.")
    nn = int(n)
    hh = float(h)
    if nn < 2 or hh <= 0:
        raise ValueError(f"need n >= 2 and h > 0, got {(nn, hh)}.")
    vw = 1.0 / (2.0 * np.sqrt(np.pi)) if VW is None else float(VW)
    gp = np.broadcast_to(np.asarray(g_prime, dtype=float), tv.shape)
    fx = np.broadcast_to(np.asarray(f_X, dtype=float), tv.shape)
    b4 = 2.0 * Sb - S * m ** 2
    b5 = gp * fx * m ** 2
    bias1 = bias2 = None
    if b2 is not None:
        c2 = np.atleast_1d(np.asarray(b2, dtype=float)).ravel()
        bias1 = hh ** 2 / (2 * S) * (c2 + m * c1) * float(mu2)
    if b3 is not None:
        c3 = np.atleast_1d(np.asarray(b3, dtype=float)).ravel()
        bias2 = hh ** 2 / (2 * S) * (c3 + m * c1) * float(mu2)
    var = b4 / (nn * S ** 2) - hh * b5 / (nn * S ** 2) * vw
    return RichResult(payload={
        "t": tv, "bias_1": bias1, "bias_2": bias2, "variance": var,
        "b4": b4, "b5": b5,
        "differ_only_in": "b_2 for the first estimator, b_3 for the second; "
                          "everything else is shared (Remark 4.3)",
        "variance_leading_order": "O(1/n); the bandwidth enters only at O(h/n), "
                                  "so the variance is far less bandwidth-sensitive "
                                  "than the bias",
        "n": nn, "h": hh,
        "method": "Theorem 4.3 (4.25)-(4.28): biases and variances of the boundary-free MRL estimators"})


def cheatsheet():
    return "fzt43: the two estimators differ ONLY by b_2 vs b_3; variance is O(1/n) with h only at O(h/n)"


#: Catalogue alias for :func:`fauzi_theorem_4_3`.
fauzi_thm4_3_mrl_bias_var = fauzi_theorem_4_3


# compact alias per ledger/NAMING.md
fauzitheorem43 = fauzi_theorem_4_3
