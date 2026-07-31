# SPDX-License-Identifier: AGPL-3.0-or-later
"""Non-separable spatio-temporal covariance functions."""

import numpy as np

from ._richresult import RichResult
from ._schab_st import (bivariate_power_mixture_correlation,
                        gneiting_covariance, gneiting_with_temporal,
                        is_valid_covariance, jones_zhang_covariance,
                        power_mixture_correlation, scale_mixture_covariance,
                        separability_test)

__all__ = ["schabenberger_st_cov_nonsep"]

_METHODS = ("monotone", "power_mixture", "scale_mixture", "differential")


def schabenberger_st_cov_nonsep(spatial_h, temporal_u, params=None,
                                method="monotone", coords=None, times=None):
    """Non-separable spatio-temporal covariance, Sec. 9.3.

    Separable models cannot represent space-time INTERACTION: under product
    separability the spatial covariance has the same shape at every time lag.
    Sec. 9.3 names four constructions that can, and all four are implemented
    here rather than one standing in for the rest, because they are not
    interchangeable.

    ``monotone`` -- Gneiting (2002), Sec. 9.3.1, eqs (9.7)-(9.9).
        C(h,k) = sigma^2 psi(|k|^2)^{-d/2} phi(||h||^2 / psi(|k|^2)) with
        phi completely monotone and psi positive with completely monotone
        derivative. With phi(t) = exp{-c t^gamma} and psi(t) = (a t^alpha + 1)^beta
        this is the closed form (9.8). Requires c, a > 0, 0 < gamma, alpha <= 1
        and 0 <= beta <= 1; outside those bounds the construction carries no
        validity guarantee, so they are enforced.

        This is the only one of the four that gives a TEST for separability:
        (9.9) is separable at beta = 0 and non-separable otherwise, and the
        two are nested. Pass ``neg2_loglik`` and ``neg2_loglik_separable`` to
        get the likelihood-ratio test with the Self and Liang (1987)
        boundary correction, which the text calls for because the null value
        sits on the edge of the parameter space.

    ``power_mixture`` -- Ma (2002), Sec. 9.3.3, eqs (9.13)-(9.14).
        Mixing product correlation functions over a discrete distribution.
        The univariate case is exactly the probability generating function of
        the mixing law evaluated at w = Rs(h) Rt(k), so a pgf is all that is
        needed; Example 9.1 gives the Binomial and Poisson cases.

    ``scale_mixture`` -- Ma (2002), Sec. 9.3.3, eqs (9.15)-(9.16).
        Z(s,t) = Zs(sU) Zt(tV): the coordinates themselves are randomly
        rescaled, so C(h,k) = integral Cs(hu) Ct(kv) dF(u,v).

    ``differential`` -- Jones and Zhang (1997), Sec. 9.3.4, eq (9.17).
        The covariance implied by a stochastic partial differential equation,
        obtained as a zero-order Hankel transform. p governs the smoothness
        of the process and must exceed max{1, d/2}. The transform is computed
        by panel quadrature and the result carries its own truncation
        diagnostics, because the integrand decays only algebraically at
        k = 0 -- a fixed cutoff here silently returns a wrong number.

    The Cressie and Huang (1999) spectral route of Sec. 9.3.2 is deliberately
    NOT offered as a black box. It requires choosing R(omega, k) and k(omega)
    and then integrating (9.12), and the text records that Gneiting (2002)
    showed some of the published examples are invalid because R did not
    satisfy the needed conditions. Constructions whose validity cannot be
    checked from the arguments alone do not belong behind a keyword; use
    ``monotone``, which was designed to avoid the spectral domain entirely.

    Parameters
    ----------
    spatial_h, temporal_u : array-like
        Spatial and temporal lags, carried separately.
    params : mapping, optional
        Method parameters. See the method descriptions above.
    method : {"monotone", "power_mixture", "scale_mixture", "differential"}
    coords, times : array-like, optional
        Design on which to check eq (9.5) numerically.

    Returns
    -------
    RichResult
        Keys: ``st_covariance``, ``method``, ``separable``, method-specific
        diagnostics, and when a design is supplied ``valid`` and
        ``min_eigenvalue``.

    References
    ----------
    Schabenberger & Gotway (2005), Sec. 9.3, eqs (9.7)-(9.17).
    """
    if method not in _METHODS:
        raise ValueError(f"`method` must be one of {_METHODS}, got {method!r}")
    p = dict(params or {})
    payload = {"method": method, "separable": False}
    lines = [("method", method)]

    if method == "monotone":
        beta = float(p.get("beta", 1.0))
        kw = dict(sigma2=p.get("sigma2", 1.0), a=p.get("a", 1.0),
                  c=p.get("c", 1.0), alpha=p.get("alpha", 1.0), beta=beta,
                  gamma=p.get("gamma", 1.0), d=p.get("d", 2))
        if "beta_t" in p:
            cov = gneiting_with_temporal(spatial_h, temporal_u,
                                         beta_t=float(p["beta_t"]), **kw)
            model = lambda d, u: gneiting_with_temporal(
                d, u, beta_t=float(p["beta_t"]), **kw)
            payload["equation"] = "9.9"
        else:
            cov = gneiting_covariance(spatial_h, temporal_u, **kw)
            model = lambda d, u: gneiting_covariance(d, u, **kw)
            payload["equation"] = "9.8"
        payload["separable"] = bool(beta == 0.0)
        lines.append(("beta", beta))
        lines.append(("separable (beta = 0)", payload["separable"]))
        if "neg2_loglik" in p and "neg2_loglik_separable" in p:
            payload["separability_test"] = separability_test(
                float(p["neg2_loglik"]), float(p["neg2_loglik_separable"]))
            lines.append(("separability p (Self-Liang)",
                          payload["separability_test"]["p_value"]))

    elif method == "power_mixture":
        rs = np.asarray(p["rs"], dtype=float)
        rt = np.asarray(p["rt"], dtype=float)
        if "pmf" in p:
            cov = bivariate_power_mixture_correlation(rs, rt, p["pmf"])
            payload["equation"] = "9.13"
        else:
            dist = p.get("distribution", "poisson")
            extra = {kk: p[kk] for kk in ("lam", "n", "pi") if kk in p}
            cov = power_mixture_correlation(rs, rt, dist, **extra)
            payload["equation"] = "9.14"
            payload["distribution"] = dist
            lines.append(("mixing law", dist))
        model = None

    elif method == "scale_mixture":
        cov = scale_mixture_covariance(spatial_h, temporal_u,
                                       p["cov_spatial"], p["cov_temporal"],
                                       p["nodes"], p["weights"])
        model = lambda d, u: scale_mixture_covariance(
            d, u, p["cov_spatial"], p["cov_temporal"], p["nodes"],
            p["weights"])
        payload["equation"] = "9.16"

    else:                                              # differential
        kw = dict(sigma2=p.get("sigma2", 1.0), theta=p.get("theta", 1.0),
                  c=p.get("c", 1.0), p=p.get("p", 1.5), d=p.get("d", 2),
                  n_quad=p.get("n_quad", 40))
        cov, meta = jones_zhang_covariance(spatial_h, temporal_u, **kw)
        model = lambda d, u: jones_zhang_covariance(d, u, **kw)[0]
        payload["equation"] = "9.17"
        payload["quadrature"] = meta
        lines += [("smoothness p", kw["p"]),
                  ("quadrature reached tau", meta["upper_reached"]),
                  ("last panel / total", meta["last_panel_rel"]),
                  ("analytic tail bound", meta["tail_bound"])]

    payload["st_covariance"] = cov

    if coords is not None and times is not None and model is not None:
        v = is_valid_covariance(coords, times, model)
        payload.update(valid=v["valid"], min_eigenvalue=v["min_eigenvalue"])
        lines += [("positive definite", v["valid"]),
                  ("min eigenvalue", v["min_eigenvalue"])]
        if not v["valid"]:
            payload["warning"] = (
                "eq (9.5) fails on this design -- the construction is not a "
                "valid covariance function here; cf. Gneiting (2002) on "
                "Cressie and Huang (1999)")

    return RichResult(title="Non-separable spatio-temporal covariance",
                      summary_lines=lines, payload=payload)


def cheatsheet():
    return ("spstcn: non-separable spatio-temporal covariance (Sec. 9.3) -- "
            "Gneiting monotone, Ma power/scale mixtures, Jones-Zhang SPDE")
