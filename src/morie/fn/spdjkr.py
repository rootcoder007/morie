# SPDX-License-Identifier: AGPL-3.0-or-later
"""Disjunctive kriging via the Chebyshev-Hermite expansion."""

from . import _array_core as np

from ._richresult import RichResult
from ._schab_hermite import (disjunctive_kriging, hermite_coefficients,
                             indicator_coefficients)

__all__ = ["schabenberger_disjunctive_kriging"]


def schabenberger_disjunctive_kriging(coords, z, target, phi_func=None,
                                      cov_model=None, degree=8,
                                      indicator_threshold=None):
    """Predict g(Z(s0)) by disjunctive kriging, Sec. 5.6.4.

    Matheron's (1976) method expands g in the Chebyshev-Hermite system,
    which is orthonormal under the standard Gaussian density, so each
    component can be kriged separately without modelling cross-covariances
    between indicators. The engine is in ``_schab_hermite``; this is the
    entry point.

    Two ways to say what is being predicted:

    * ``phi_func`` -- any smooth g. Its coefficients (5.65) are found by
      Gauss-Hermite quadrature, which is exact for polynomials.
    * ``indicator_threshold`` -- g(Z) = I(Z <= z_k). Its coefficients come
      from the CLOSED FORM (5.72), not from quadrature. Quadrature is exact
      for polynomials and the indicator is a step function, so it converges
      slowly and silently: at degree 6 it returns b_0 = 0.683 where
      F(z_k) = 0.758. Since the indicator is the canonical target of the
      method, getting that wrong would be wrong in exactly the case
      disjunctive kriging exists for.

    Parameters
    ----------
    coords : array-like, shape (n, d)
        Sampling locations.
    z : array-like, shape (n,)
        Data on the GAUSSIAN scale. If the data are not Gaussian, transform
        them first with the normal-scores transform of Sec. 5.6.2
        (``morie.fn.sptgk.normal_scores``), which is what Sec. 5.6.4.3
        prescribes.
    target : array-like, shape (d,)
        Prediction location.
    phi_func : callable, optional
        The function g to predict. Defaults to the identity.
    cov_model : callable, optional
        The CORRELATION function rho(h). Matheron's result gives
        Cov[eta_p(Z(s+h)), eta_p(Z(s))] = rho(h)^p, so each component is
        kriged with rho raised to its own power.
    degree : int
        How many Hermite terms. The text notes rho(h)^p tends to white noise
        as p grows, so "in practice only a few (usually less than a dozen
        ...) Hermite polynomials need to be predicted"; the default of 8 sits
        inside that guidance.
    indicator_threshold : float, optional
        If given, predict I(Z <= z_k) instead of ``phi_func``.

    Returns
    -------
    RichResult
        Keys: ``prediction``, ``variance``, ``coefficients``,
        ``component_variances``, ``degree``.

    References
    ----------
    Schabenberger Ch 5, Sec 5.6.4
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    z = np.asarray(z, dtype=float).ravel()
    target = np.asarray(target, dtype=float).ravel()
    degree = int(degree)
    if cov_model is None:
        def cov_model(h):
            return np.exp(-np.asarray(h, dtype=float))

    if indicator_threshold is not None:
        b = indicator_coefficients(indicator_threshold, degree)
        label = "indicator I(Z <= %g)" % float(indicator_threshold)

        def g(_x, _b=b):
            raise RuntimeError("coefficients supplied directly")
        pred, var, _, comp = _predict_with_coefficients(
            coords, z, target, cov_model, b, degree)
    else:
        if phi_func is None:
            def phi_func(v):
                return v
        pred, var, b, comp = disjunctive_kriging(
            coords, z, target, cov_model, phi_func, degree=degree)
        label = "g supplied as phi_func"

    return RichResult(
        title="Disjunctive kriging",
        summary_lines=[("prediction", pred), ("variance", var),
                       ("Hermite degree", degree), ("target", label)],
        payload={"prediction": float(pred), "variance": float(var),
                 "coefficients": b, "component_variances": comp,
                 "degree": degree, "method": "disjunctive kriging"},
    )


def _predict_with_coefficients(coords, z, target, correlation_fn, b, degree):
    """The (5.67)-(5.71) loop for coefficients already in hand."""
    from ._schab_hermite import hermite_orthonormal
    d_mat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    d_vec = np.linalg.norm(coords - target, axis=1)
    rho_mat = np.asarray(correlation_fn(d_mat), dtype=float)
    rho_vec = np.asarray(correlation_fn(d_vec), dtype=float)
    eta_data = hermite_orthonormal(z, degree)
    pred = float(b[0])
    var = 0.0
    comp = np.zeros(degree + 1)
    for p in range(1, degree + 1):
        r_mat = rho_mat**p
        r_vec = rho_vec**p
        try:
            lam = np.linalg.solve(r_mat, r_vec)
        except np.linalg.LinAlgError:
            lam = np.linalg.lstsq(r_mat, r_vec, rcond=None)[0]
        pred += float(b[p]) * float(lam @ eta_data[p])
        s2 = 1.0 - float(lam @ r_vec)
        comp[p] = s2
        var += float(b[p]) ** 2 * s2
    return pred, var, b, comp


def cheatsheet():
    return "spdjkr: disjunctive kriging, eqs (5.64)-(5.72) (Schabenberger Sec 5.6.4)"
