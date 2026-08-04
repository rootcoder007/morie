# morie.fn -- function file (rootcoder007/morie)
"""Kernel density estimator with Silverman's window-width rules."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wasserman_kde"]


def wasserman_kde(x, data, h=None, rule="3.31"):
    r"""Kernel density estimate, Silverman Eq. (2.2a):

    .. math:: \hat f(x) = \frac1{nh}\sum_{i=1}^n
              K\!\left(\frac{x - X_i}{h}\right).

    ``h`` is the WINDOW WIDTH -- Silverman's term; "smoothing
    parameter" and "bandwidth" are the same thing. Because ``K`` here
    is a probability density, non-negative and integrating to one,
    the estimate is itself a probability density and inherits every
    continuity and differentiability property of ``K``.

    The default window width is Silverman's own recommendation,
    Eq. (3.31),

    .. math:: h = 0.9\,A\,n^{-1/5},\qquad
              A = \min(\hat\sigma, R/1.34),

    with :math:`R` the interquartile range (3.30). **Not**
    :math:`1.06\hat\sigma n^{-1/5}`: that is Eq. (3.28), the pure
    normal reference, which the book presents as a starting point and
    then improves on twice. (3.28) oversmooths long-tailed, skewed
    and bimodal data, and the adaptive spread ``A`` is what fixes it
    -- a single outlier moves :math:`\hat\sigma` a long way and the
    interquartile range hardly at all.

    Silverman reports that (3.31) lands within 10% of the optimal
    mean integrated square error for every t-distribution he
    considers, for log-normals with skewness up to about 1.8, and for
    normal mixtures separated by up to 3 standard deviations.

    Parameters
    ----------
    x : array-like
        Evaluation points.
    data : array-like
        Sample.
    h : float, optional
        Window width; ``rule`` decides it when omitted.
    rule : {"3.31", "3.28", "3.29"}, default "3.31"
        Which of the book's rules to apply.

    Returns
    -------
    RichResult
        keys: ``x``, ``density``, ``h``, ``rule``, ``adaptive_spread``,
        ``h_normal_reference`` (3.28), ``h_iqr`` (3.29), ``mass``,
        ``is_density``, ``n``, ``method``.

    References
    ----------
    Silverman, B. W. (1986), *Density Estimation for Statistics and
    Data Analysis*, Chapman and Hall. Eq. (2.2a) for the estimator,
    Sec. 3.4.2 Eqs. (3.28)-(3.31) for the window width. Read from the
    PDF. Rosenblatt (1956); Parzen (1962).
    """
    from ._wsm import adaptive_spread, silverman_bandwidth

    d = np.asarray(data, dtype=float).ravel()
    n = d.size
    if n < 2:
        raise ValueError(f"need at least 2 observations, got {n}.")
    hh = silverman_bandwidth(d, rule) if h is None else float(h)
    if hh <= 0:
        raise ValueError(f"the window width must be positive, got {hh}.")
    g = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    u = (g[:, None] - d[None, :]) / hh
    dens = np.exp(-0.5 * u ** 2).sum(axis=1) / (n * hh * np.sqrt(2 * np.pi))
    mass = None
    if g.size > 2 and np.all(np.diff(g) > 0):
        mass = float(np.trapezoid(dens, g))
    return RichResult(payload={
        "x": g, "density": dens, "h": hh, "rule": rule,
        "adaptive_spread": adaptive_spread(d),
        "h_normal_reference": silverman_bandwidth(d, "3.28"),
        "h_iqr": silverman_bandwidth(d, "3.29"),
        "mass": mass, "is_density": True,
        "why_not_1_06": "1.06 sigma n^(-1/5) is (3.28), the pure normal "
                        "reference; (3.31) replaces sigma with the adaptive "
                        "spread A and the constant with 0.9, and that is what "
                        "the book actually recommends",
        "n": int(n),
        "method": "Silverman (2.2a) kernel density estimate, window width by (3.31)"})


def cheatsheet():
    return "wsmkdn: Silverman's rule is 0.9 A n^(-1/5), not 1.06 sigma n^(-1/5)"


# compact alias per ledger/NAMING.md
wassermankde = wasserman_kde
