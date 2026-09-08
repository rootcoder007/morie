# morie.fn -- function file (rootcoder007/morie)
"""Proportional hazards model with unobserved heterogeneity: nonparametric frailty."""

from . import _array_core as np
from . import _horowitz as HZ
from . import _hrz3 as H
from . import _s03core as core

from ._richresult import RichResult
from .hrztf import horowitz_both_nonpar_transform
from .hrztmod import horowitz_transformation_model

__all__ = ["horowitz_ph_frailty_nonpar"]


def horowitz_ph_frailty_nonpar(t, x, event=None, ny=21, nz=21, nq=21,
                               q=0.22, delta=0.85, bandwidth=None):
    r"""Proportional hazards model with unobserved heterogeneity, with
    the baseline hazard AND the frailty distribution both
    nonparametric.

    Horowitz (2009), Section 6.3.4, pages 223-226, implementing
    Horowitz (1999).  The model
    :math:`\lambda(y|x, v) = \lambda_0(y)\exp(-x'\beta)v`
    is equivalent to

    .. math:: \log\Lambda_0(Y) = X'\beta + V + U                \quad (6.68)

    with :math:`U` independent of :math:`(X, V)` and
    :math:`F(u) = 1 - \exp(-e^{-u})`.  Writing the transformation
    model as :math:`T(Y) = X'\alpha + W` (6.69) relates the two by
    :math:`T(y) = \sigma^{-1}\log\Lambda_0(y)` and
    :math:`W = (V+U)/\sigma` with :math:`\sigma = |\beta_1|`, so
    (6.68) is a RESCALED transformation model and everything reduces
    to estimating the single scalar :math:`\sigma`.

    :math:`\sigma` is recovered from the small-:math:`y` limit

    .. math:: \sigma(y) = \frac{\int G_z(y|z)p_Z(z)^2\,dz}
                               {\int G(y|z)p_Z(z)^2\,dz},
              \qquad \sigma = \lim_{y\to 0}\sigma(y)      \quad (6.75),(6.76)

    whose sample analogue carries an explicit leading minus:

    .. math:: \sigma_n(y) = -\frac{\int G_{nz}(y|z)p_{nZ}(z)^2\,dz}
                                  {\int G_n(y|z)p_{nZ}(z)^2\,dz}
                                                              \quad (6.80)

    That minus sign was read off a RENDERED IMAGE of page 225, not an
    extracted text layer, because ``pdftotext`` drops minus signs.  It
    is load-bearing: :math:`G_z < 0` by (6.74), so without it
    :math:`\sigma_n` comes out negative and every downstream quantity
    inverts.  Note also that (6.80) uses the kernel argument
    :math:`(z - Z_{nj})/h_n`, the OPPOSITE orientation to (6.61) in
    Section 6.3.1; each section is implemented as printed, which is
    why the two derivative terms differ in sign here and in
    ``hrztmod``.

    The plain estimator :math:`\sigma_n(y_n)` converges no faster than
    :math:`n^{-1/3}`.  Under :math:`Ee^{-3V} < \infty` (PHU3(ii)) the
    Schucany-Sommers bias correction

    .. math:: \sigma_n = \frac{\sigma_n(y_{n1})
              - n^{-q(1-\delta)}\sigma_n(y_{n2})}{1 - n^{-q(1-\delta)}}
                                                              \quad (6.81)

    reaches a rate arbitrarily close to the :math:`n^{-2/5}` that
    Ishwaran (1996) shows is optimal, and that correction is applied.
    The admissible ranges :math:`1/5 < q < 1/4` and
    :math:`1/(2q) - 3/2 < \delta < 1` are enforced.

    :math:`y_{n1}, y_{n2}` are defined by
    :math:`\Lambda_0(y_{n1}) = cn^{-q}` and
    :math:`\Lambda_0(y_{n2}) = cn^{-\delta q}`, which is circular
    since :math:`\Lambda_0` is what is being estimated.  For small
    :math:`y`, :math:`P(Y \le y) = 1-\int e^{-\Lambda_0(y)e^{-v}}dF_V
    \approx \Lambda_0(y)Ee^{-V}`, so the CDF level is proportional to
    :math:`\Lambda_0`; the two points are therefore taken as the
    empirical quantiles of :math:`Y` at levels :math:`n^{-q}` and
    :math:`n^{-\delta q}`.  This resolution of the circularity is
    stated here because the text does not give one -- p. 173 notes
    that "methods for choosing a_n and h_n in applications are not yet
    available", and the same is true of these sequences.

    Then :math:`\Lambda_0` and :math:`\lambda_0` follow from

    .. math:: \Lambda_{n0}(y) = \exp[\sigma_n T_n(y)]           \quad (6.70)

    .. math:: \lambda_{n0}(y) = \sigma_n T_n'(y)
              \exp[\sigma_n T_n(y)]                             \quad (6.71)

    and :math:`\beta = \sigma\alpha`.  Because :math:`T_n(y_0) = 0` by
    construction, :math:`\Lambda_{n0}(y_0) = 1` exactly, which is the
    location normalisation the section requires.

    ``frailty_dist`` is the estimated CDF of :math:`W = (V+U)/\sigma`
    from (6.66), which is what the data identify without a further
    deconvolution step; recovering :math:`F_V` itself would require
    deconvolving the known extreme-value :math:`F_U` out of it, and
    that is NOT done here.  Only Elbers-Ridder identification
    (:math:`Ee^{-V} < \infty`) plus PHU3(ii) is assumed.

    The stub docstring this replaced made three claims that Section
    6.3.4 contradicts, all checked against pages 223-225:

    * "identification via multiple spells".  The section identifies
      from a SINGLE spell: "Elbers and Ridder (1982) showed that model
      (6.68) is identified if Ee^{-V} < infinity" (p. 223).  The word
      "spell" does not occur in the section.  Multiple-spell
      identification is a different literature and is not used here.
    * "V arbitrary with E[V] = 1".  The normalisation the section
      actually imposes is Lambda_0(y_0) = 1 for some finite y_0 > 0
      (p. 223), together with |alpha_1| = 1 on the index.  E[V] = 1 is
      never assumed; what is assumed is the moment condition
      Ee^{-3V} < infinity (PHU3(ii)).
    * ``h(t|X,V) = h_0(t) exp(X'beta) V``.  Equation (6.72) writes the
      hazard as lambda(y|z,v) = lambda_0(y) exp[-(sigma z + v)], i.e.
      with a NEGATIVE index and an exp(-v) frailty.  The published
      parameterisation is the one implemented.

    The estimator below follows the source, not the stub.

    Parameters
    ----------
    t : array-like, shape (n,)
        Observed durations; must be strictly positive.
    x : array-like, shape (n,) or (n, d)
        Covariates.  The first column carries the normalisation.
    event : array-like of {0, 1}, optional
        Censoring indicator; 1 = observed failure.  Censored rows are
        dropped, since (6.73)-(6.80) are written for observed Y.
        Default: all observed.
    ny, nz : int
        Grid sizes for the transformation-model step.
    nq : int, default 21
        Quadrature points for the integrals in (6.80).
    q : float, default 0.22
        Rate constant, must lie strictly in (1/5, 1/4).
    delta : float, default 0.85
        Bias-correction constant, must lie in (1/(2q) - 3/2, 1).
    bandwidth : float, optional
        Common bandwidth; default Silverman's rule.

    Returns
    -------
    RichResult
        keys: ``beta_hat``, ``alpha_hat``, ``sigma``, ``sigma_y1``,
        ``sigma_y2``, ``h0_hat`` (lambda_0 on ``y_grid``),
        ``Lambda0_hat``, ``frailty_dist``, ``frailty_grid``,
        ``T_hat``, ``y_grid``, ``y0``, ``n``, ``n_used``, ``method``.

    References
    ----------
    Horowitz, J. L. (2009). *Semiparametric and Nonparametric Methods
    in Econometrics*. Springer, Sec. 6.3.4, eqs. (6.68)-(6.81),
    pp. 223-226.
    Horowitz, J. L. (1999). Semiparametric estimation of a
    proportional hazard model with unobserved heterogeneity.
    *Econometrica* 67(5), 1001-1028.
    Elbers, C. & Ridder, G. (1982). True and spurious duration
    dependence. *Review of Economic Studies* 49(3), 403-409.
    Ishwaran, H. (1996). Uniform rates of estimation in the
    semiparametric Weibull mixture model.
    *Annals of Statistics* 24(4), 1572-1585.
    """
    tv = np.asarray(t, dtype=float).ravel()
    n_all = int(tv.size)
    X = np.atleast_2d(np.asarray(x, dtype=float))
    if X.shape[0] != n_all and X.shape[1] == n_all:
        X = X.T
    if X.shape[0] != n_all:
        raise ValueError(f"x must have {n_all} rows, got shape {X.shape}.")
    d = int(X.shape[1])
    for i in range(n_all):
        if not (float(tv[i]) > 0):
            raise ValueError(
                f"durations must be strictly positive; t[{i}] = {float(tv[i])}.")
    if event is None:
        keep = list(range(n_all))
    else:
        ev = np.asarray(event, dtype=float).ravel()
        if ev.size != n_all:
            raise ValueError(
                f"t has {n_all} points but event has {ev.size}.")
        keep = [i for i in range(n_all) if float(ev[i]) != 0.0]
    n = len(keep)
    if n < 10:
        raise ValueError(
            f"need at least 10 uncensored observations, got {n}.")
    q = float(q)
    delta = float(delta)
    if not (0.2 < q < 0.25):
        raise ValueError(f"q must lie strictly in (1/5, 1/4), got {q}.")
    lo_d = 1.0 / (2.0 * q) - 1.5
    if not (lo_d < delta < 1.0):
        raise ValueError(
            f"delta must lie in ({lo_d:.6g}, 1); got {delta}.")
    nq = int(nq)
    if nq < 3:
        raise ValueError(f"nq must be at least 3, got {nq}.")

    yv = [float(tv[i]) for i in keep]
    Xk = [[float(X[i][k]) for k in range(d)] for i in keep]

    hb = float(bandwidth) if bandwidth is not None else HZ.silverman_bw(
        [Xk[i][0] for i in range(n)])
    alpha = H.index_dir(Xk, yv, hb)
    Z = [sum(Xk[i][k] * float(alpha[k]) for k in range(d)) for i in range(n)]
    hz = float(bandwidth) if bandwidth is not None else HZ.silverman_bw(Z)

    za = core.quantile7(Z, 0.05)
    zb = core.quantile7(Z, 0.95)
    if not (zb > za):
        raise ValueError("the index has no spread.")
    dz = (zb - za) / (nq - 1)
    zg = [za + k * dz for k in range(nq)]
    wq = [dz] * nq
    wq[0] = dz / 2.0
    wq[nq - 1] = dz / 2.0

    def sigma_at(yy):
        """(6.80) with the kernel argument (z - Z_nj)/h as printed."""
        num = den = 0.0
        for k in range(nq):
            z = zg[k]
            A = B = Az = Bz = 0.0
            for i in range(n):
                u = (z - Z[i]) / hz
                kk = np.exp(-0.5 * u * u) / H.SQRT2PI
                dk = -(u / hz) * kk        # d/dz K((z - Z_i)/h)
                ind = 1.0 if yv[i] <= yy else 0.0
                A += ind * kk
                B += kk
                Az += ind * dk
                Bz += dk
            if B <= 1e-300:
                continue
            pnz = B / (n * hz)
            Gn = A / B
            Gnz = (Az * B - A * Bz) / (B * B)
            num += wq[k] * Gnz * pnz * pnz
            den += wq[k] * Gn * pnz * pnz
        if abs(den) < 1e-300:
            raise ValueError(
                "the denominator of (6.80) vanished; y is too small for the "
                "sample to identify sigma.")
        return -num / den

    yn1 = core.quantile7(yv, min(n ** (-q), 0.99))
    yn2 = core.quantile7(yv, min(n ** (-delta * q), 0.99))
    s1 = sigma_at(yn1)
    s2 = sigma_at(yn2)
    fac = n ** (-q * (1.0 - delta))
    if abs(1.0 - fac) < 1e-12:
        raise ValueError("the bias-correction weight in (6.81) is degenerate.")
    sigma = (s1 - fac * s2) / (1.0 - fac)

    tf = horowitz_both_nonpar_transform(Xk, yv, ny=ny, nz=nz,
                                        bandwidth=bandwidth).payload
    T = tf["T_hat"]
    yg = tf["y_grid"]
    m = len(yg)
    dv = yg[1] - yg[0]
    Tp = [0.0] * m
    for k in range(m):
        if k == 0:
            Tp[k] = (T[1] - T[0]) / dv
        elif k == m - 1:
            Tp[k] = (T[m - 1] - T[m - 2]) / dv
        else:
            Tp[k] = (T[k + 1] - T[k - 1]) / (2.0 * dv)

    Lam = [np.exp(sigma * T[k]) for k in range(m)]           # (6.70)
    lam = [sigma * Tp[k] * np.exp(sigma * T[k]) for k in range(m)]  # (6.71)
    beta = [sigma * float(alpha[k]) for k in range(d)]

    return RichResult(payload={
        "beta_hat": beta,
        "alpha_hat": [float(t) for t in alpha],
        "sigma": float(sigma),
        "sigma_y1": float(s1),
        "sigma_y2": float(s2),
        "h0_hat": lam,
        "Lambda0_hat": Lam,
        "frailty_dist": tf["F_hat"],
        "frailty_grid": tf["u_grid"],
        "T_hat": T,
        "y_grid": yg,
        "y0": tf["y0"],
        "n": n_all,
        "n_used": n,
        "method": "Horowitz (2009) eqs. (6.80)-(6.81), (6.70)-(6.71)",
    })


def cheatsheet():
    return "hrzphvnp: (6.80) carries a leading minus; Lambda0(y0)=1 exactly"
