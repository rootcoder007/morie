# morie.fn -- function file (rootcoder007/morie)
"""Tau-estimator regression (Yohai and Zamar 1988)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["tau_regression"]


def tau_regression(X, y, n_subsets=200, seed=0, c1=1.5476, c2=6.08):
    r"""The tau-estimator of Yohai and Zamar (1988): minimise the
    TAU-SCALE of the residuals,

    .. math:: \tau^2(\beta) = s(\beta)^2 \cdot \frac1{n b_2}\sum_i
              \rho_2\!\left(\frac{r_i(\beta)}{s(\beta)}\right),

    where :math:`s(\beta)` is the M-scale under a tight
    :math:`\rho_1` (biweight, :math:`c_1 = 1.5476`, giving the 50%
    breakdown) and :math:`\rho_2` is a wide biweight
    (:math:`c_2 = 6.08`, whose calibration gives roughly 95% normal
    efficiency). The construction differs from MM in WHERE the
    efficiency lives: MM freezes an S-scale and re-fits beta, while
    the tau-estimator bakes both rhos into ONE objective, so the
    minimiser is simultaneously the location of high breakdown and
    high efficiency, and the tau-scale itself is a robust,
    efficiency-calibrated residual scale -- useful as a number in its
    own right.

    Computation is the same random p-subset strategy as the
    S-estimator, ranking subsets by :math:`\tau` rather than by
    :math:`s`, with local IRLS refinement under the combined weight
    function of their Sec. 4.

    Parameters
    ----------
    x, y : array-like
        Design (constant added when absent) and response.
    n_subsets : int, default 200
        Random p-subsets.
    seed : int, default 0
        Subset seed.
    c1, c2 : float
        The two biweight constants; the defaults are the paper's.

    Returns
    -------
    RichResult
        keys: ``beta``, ``tau_scale``, ``m_scale``, ``residuals``,
        ``breakdown``, ``gaussian_efficiency``, ``c1``, ``c2``,
        ``versus_mm``, ``n``, ``p``, ``method``.

    References
    ----------
    Yohai, V. J. and Zamar, R. H. (1988), "High breakdown-point
    estimates of regression by means of the minimization of an
    efficient scale", *JASA* 83:406-413, Secs. 2 and 4.
    """
    from scipy import integrate, stats

    from ._robust import prepare_design, s_scale, tukey_rho, tukey_weight

    A, yv = prepare_design(X, y)
    n, p = A.shape
    if n <= p:
        raise ValueError(f"need more observations than parameters, "
                         f"got n = {n}, p = {p}.")
    c1 = float(c1)
    c2 = float(c2)
    # b2 = E_Phi[rho_2], the consistency constant for the tau-scale
    b2, _ = integrate.quad(lambda u: tukey_rho(u, c2) * stats.norm.pdf(u),
                           -12, 12)

    def tau_of(beta):
        r = yv - A @ beta
        s = s_scale(r, c=c1, b=0.5)
        if s <= 0:
            return 0.0, 0.0
        t2 = s ** 2 * float(np.mean(tukey_rho(r / s, c2))) / b2
        return float(np.sqrt(max(t2, 0.0))), s

    rng = np.random.default_rng(seed)
    best = (np.inf, None, 0.0)
    for _ in range(int(n_subsets)):
        idx = rng.choice(n, p, replace=False)
        sub = A[idx]
        if np.linalg.matrix_rank(sub) < p:
            continue
        try:
            beta = np.linalg.solve(sub, yv[idx])
        except np.linalg.LinAlgError:
            continue
        t, s = tau_of(beta)
        if 0 < t < best[0]:
            best = (t, beta, s)
    if best[1] is None:
        raise ValueError("no non-singular p-subset was found.")
    tau, beta, s = best[0], best[1], best[2]
    # local refinement with the combined weight of Sec. 4
    for _ in range(50):
        r = yv - A @ beta
        s = s_scale(r, c=c1, b=0.5)
        if s <= 0:
            break
        u = r / s
        w = tukey_weight(u, c1) + tukey_weight(u, c2)
        Aw = A * w[:, None]
        beta_new = np.linalg.lstsq(Aw.T @ A, Aw.T @ yv, rcond=None)[0]
        t_new, _ = tau_of(beta_new)
        if t_new >= tau - 1e-12:
            break
        tau, beta = t_new, beta_new
    return RichResult(payload={
        "beta": beta, "tau_scale": float(tau), "m_scale": float(s),
        "residuals": yv - A @ beta,
        "breakdown": 0.5, "gaussian_efficiency": 0.95,
        "c1": c1, "c2": c2,
        "versus_mm": "MM freezes an S-scale and re-fits beta; the "
                     "tau-estimator bakes both rhos into ONE objective, and "
                     "the tau-scale is itself a robust efficient residual "
                     "scale",
        "n": int(n), "p": int(p),
        "method": "Tau-estimator (Yohai-Zamar 1988): minimise the efficient "
                  "tau-scale, c1 = 1.5476, c2 = 6.08"})


def cheatsheet():
    return "taubrg: one objective carries both rhos -- the tau-scale is efficient AND 50%-breakdown"
