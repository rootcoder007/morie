# morie.fn -- shared engine (rootcoder007/morie)
"""Discrete-time survival TMLE: hazard fits, targeting, influence curve.

The estimator is the hazard-based TMLE of the treatment-specific survival
curve. Every formula below is taken from Cai and van der Laan (2020),
"One-step targeted maximum likelihood estimation for time-to-event
outcomes", *Biometrics* 76:722-733 (preprint arXiv:1802.09479), whose
equation numbers are cited at the point of use.

The likelihood factorises (their equation 1) into the confounder
distribution, the propensity score, the failure hazard and the censoring
hazard. Only the failure hazard is fluctuated -- the other three are
tangent to the parameter and are left alone.
"""

import numpy as np

from ._did import add_intercept, logit_fit, logit_predict

__all__ = [
    "discretise_times",
    "hazard_design",
    "fit_hazards",
    "survival_from_hazard",
    "target_arm",
    "survival_tmle",
]

_CLIP = 1e-12


def _logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1.0 - p))


def _expit(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def discretise_times(time, n_bins=None):
    """Map observed times onto the integer grid 1, ..., tmax.

    The hazard formulation is inherently discrete. When the observed
    times already take few distinct values, they ARE the grid and no
    information is lost. When they are continuous, coarsening is a real
    modelling choice, so the number of bins is returned alongside the
    grid rather than being hidden.

    Returns
    -------
    k : ndarray of int, shape (n,)
        Bin index in 1..tmax for each observation.
    edges : ndarray
        Upper edge of each bin on the original time scale.
    """
    t = np.asarray(time, dtype=float).ravel()
    if t.size == 0:
        raise ValueError("time is empty.")
    if np.any(~np.isfinite(t)):
        raise ValueError("time contains non-finite values.")
    if np.any(t < 0):
        raise ValueError("time must be non-negative.")
    uniq = np.unique(t)
    if n_bins is None:
        n_bins = uniq.size if uniq.size <= 50 else 20
    n_bins = int(n_bins)
    if n_bins < 2:
        raise ValueError("need at least 2 time bins, got %d." % n_bins)
    if uniq.size <= n_bins:
        edges = uniq
    else:
        qs = np.linspace(0.0, 1.0, n_bins + 1)[1:]
        edges = np.unique(np.quantile(uniq, qs))
    k = np.searchsorted(edges, t, side="left") + 1
    return k.astype(int), edges


def hazard_design(W, A, k, tmax, max_dummies=25):
    """Design matrix for the pooled hazard regression.

    Cai and van der Laan pool every person-time row into ONE model with
    the time index as a feature, noting that smoothing over time speeds
    up the fit. Time enters as dummies while the grid is short enough
    to afford one coefficient per bin, and as a cubic polynomial after
    that -- the alternative, a separate model per time, throws away the
    smoothness that makes the pooled fit work at all.

    Treatment enters with its interactions against every covariate.
    Without them the fitted hazard is forced parallel across arms, and
    the counterfactual survival curves inherit a proportional-hazards
    assumption that nothing in the method requires.
    """
    W = np.atleast_2d(np.asarray(W, dtype=float))
    if W.shape[0] == 1 and W.shape[1] != np.asarray(A).size:
        W = W.T
    A = np.asarray(A, dtype=float).ravel()
    k = np.asarray(k, dtype=float).ravel()
    cols = [np.ones(A.size), A]
    if tmax <= max_dummies:
        for j in range(2, int(tmax) + 1):
            cols.append((k == j).astype(float))
    else:
        s = (k - 1.0) / max(tmax - 1.0, 1.0)
        cols += [s, s ** 2, s ** 3]
    for j in range(W.shape[1]):
        cols.append(W[:, j])
        cols.append(A * W[:, j])
    return np.column_stack(cols)


def _long_rows(k_obs, event, tmax):
    """Person-time rows: (subject, time) pairs that are still at risk.

    A subject contributes rows k = 1, ..., min(K_i, tmax): they are at
    risk at every time up to and including the one at which they leave.
    """
    idx, tt = [], []
    for i, ki in enumerate(k_obs):
        top = min(int(ki), int(tmax))
        for j in range(1, top + 1):
            idx.append(i)
            tt.append(j)
    return np.asarray(idx, dtype=int), np.asarray(tt, dtype=int)


def fit_hazards(W, A, k_obs, event, tmax):
    """Pooled logistic fits of the failure and censoring hazards.

    Their equations (4)-(7): the failure hazard is the probability of
    ``dN(t) = 1`` given still at risk, and the censoring hazard is the
    same thing with the roles of ``N`` and ``Ac`` swapped.

    Returns
    -------
    lam1, lam0, lamc1, lamc0 : ndarray, shape (n, tmax)
        Failure and censoring hazards evaluated counterfactually at
        ``A = 1`` and ``A = 0`` for every subject's covariates.
    """
    n = len(k_obs)
    ridx, rt = _long_rows(k_obs, event, tmax)
    if ridx.size == 0:
        raise ValueError("no person-time rows; check time and tmax.")
    Wl = np.atleast_2d(np.asarray(W, dtype=float))
    if Wl.shape[0] != n:
        Wl = Wl.T
    dN = ((k_obs[ridx] == rt) & (event[ridx] == 1)).astype(float)
    dC = ((k_obs[ridx] == rt) & (event[ridx] == 0)).astype(float)

    out = []
    for y_row in (dN, dC):
        X = hazard_design(Wl[ridx], A[ridx], rt, tmax)
        beta, _ = logit_fit(X, y_row)
        arm = []
        for a in (1.0, 0.0):
            H = np.empty((n, int(tmax)))
            for j in range(1, int(tmax) + 1):
                Xj = hazard_design(
                    Wl, np.full(n, a), np.full(n, float(j)), tmax
                )
                H[:, j - 1] = logit_predict(Xj, beta)
            arm.append(np.clip(H, _CLIP, 1 - _CLIP))
        out.extend(arm)
    return out[0], out[1], out[2], out[3]


def survival_from_hazard(lam):
    r"""S(t) = prod_{k<=t} (1 - lambda(k)), Cai and van der Laan section 3."""
    return np.cumprod(1.0 - np.asarray(lam, dtype=float), axis=1)


def _clever(g_a, S_a, SC_a, t0):
    r"""Clever covariate, Cai and van der Laan equation (3).

    .. math::
       h_t(k, A, W) = -\,\frac{I(A=a)\,I(k \le t)}
                              {g(a \mid W)\, S_{A^c}(k^- \mid A, W)}
                      \,\frac{S_N(t \mid A, W)}{S_N(k \mid A, W)}.

    The sign is negative because the parameter is the SURVIVAL
    probability; targeting the cumulative incidence instead flips it.
    The ``I(A = a)`` factor is left out here and applied by the caller,
    which is what makes the same array usable both for fitting epsilon
    on the observed rows and for evaluating the update counterfactually
    at ``A = a`` for every subject.

    ``S_Ac(k^-)`` is the censoring survival just BEFORE k, i.e. through
    k-1, so the first time bin divides by one rather than by an
    already-censored quantity.
    """
    n, tmax = S_a.shape
    Sminus = np.ones_like(SC_a)
    Sminus[:, 1:] = SC_a[:, :-1]
    H = np.zeros((n, tmax))
    j = int(t0) - 1
    H[:, : j + 1] = -(
        (S_a[:, [j]] / np.maximum(S_a[:, : j + 1], _CLIP))
        / (g_a[:, None] * np.maximum(Sminus[:, : j + 1], _CLIP))
    )
    return H


def _fit_epsilon(y, offset, cov, weights, max_iter=100, tol=1e-12):
    """One-parameter logistic fluctuation, no intercept, fitted by IRLS.

    Their equation (11): ``logit(lambda*) = logit(lambda) + eps * h``,
    with the initial hazard as an offset and ``h`` the only covariate.
    """
    eps = 0.0
    for _ in range(int(max_iter)):
        p = _expit(offset + eps * cov)
        w = weights * np.maximum(p * (1 - p), 1e-10)
        score = float(np.sum(weights * cov * (y - p)))
        info = float(np.sum(w * cov * cov))
        if info <= 1e-14:
            break
        step = score / info
        eps += step
        if abs(step) < tol:
            break
        if abs(eps) > 50:
            break
    return float(eps)


def target_arm(a, A, k_obs, event, lam_a, lamc_a, g_a, t0,
               max_iter=100):
    """Iterate the targeting step for one treatment arm.

    Cai and van der Laan are explicit that a single fluctuation is not
    enough: "in practice one iteration is not enough and one might have
    to iterate many times until ||eps_n|| is small or explicitly check
    the value of (1/n) sum D*(P*)(O_i) smaller than a threshold". The
    threshold used here is 1/n, which is the convention adopted by the
    ``survtmle`` reference implementation.

    Returns
    -------
    dict with the targeted hazard, survival, plug-in estimate, influence
    curve, number of iterations and the final epsilon.
    """
    n = len(k_obs)
    tmax = lam_a.shape[1]
    j0 = int(t0) - 1
    in_arm = (A == a).astype(float)
    S_a = survival_from_hazard(lam_a)
    SC_a = survival_from_hazard(lamc_a)

    # observed person-time rows, restricted to k <= t0 and still at risk
    ridx, rt = _long_rows(k_obs, event, min(int(t0), int(tmax)))
    dN = ((k_obs[ridx] == rt) & (event[ridx] == 1)).astype(float)
    wt = in_arm[ridx]

    eps, iters, eic = 0.0, 0, None
    for iters in range(1, int(max_iter) + 1):
        H = _clever(g_a, S_a, SC_a, t0)
        # influence curve, their equation (2)
        atrisk = np.zeros((n, tmax))
        for i in range(n):
            atrisk[i, : min(int(k_obs[i]), tmax)] = 1.0
        dNmat = np.zeros((n, tmax))
        ev = (event == 1) & (k_obs <= tmax)
        dNmat[np.nonzero(ev)[0], k_obs[ev] - 1] = 1.0
        resid = dNmat[:, : j0 + 1] - atrisk[:, : j0 + 1] * lam_a[:, : j0 + 1]
        eic = in_arm * np.sum(H[:, : j0 + 1] * resid, axis=1)
        psi = float(np.mean(S_a[:, j0]))
        eic = eic + S_a[:, j0] - psi
        if abs(float(np.mean(eic))) <= 1.0 / n:
            break
        eps = _fit_epsilon(
            dN, _logit(lam_a[ridx, rt - 1]), H[ridx, rt - 1], wt
        )
        if not np.isfinite(eps) or eps == 0.0:
            break
        lam_a = lam_a.copy()
        lam_a[:, : j0 + 1] = np.clip(
            _expit(_logit(lam_a[:, : j0 + 1]) + eps * H[:, : j0 + 1]),
            _CLIP, 1 - _CLIP,
        )
        S_a = survival_from_hazard(lam_a)

    psi = float(np.mean(S_a[:, j0]))
    return {
        "hazard": lam_a,
        "survival": S_a,
        "curve": np.mean(S_a, axis=0),
        "psi": psi,
        "eic": eic,
        "epsilon": eps,
        "iterations": iters,
        "converged": bool(abs(float(np.mean(eic))) <= 1.0 / n),
    }


def survival_tmle(time, event, A, W, t0=None, n_bins=None, trunc=0.025,
                  max_iter=100):
    """Full discrete-time survival TMLE for S_1(t0), S_0(t0) and the
    difference.

    Returns a plain dict; the public wrappers dress it up.
    """
    t = np.asarray(time, dtype=float).ravel()
    ev = np.asarray(event, dtype=float).ravel()
    a = np.asarray(A, dtype=float).ravel()
    Wa = np.atleast_2d(np.asarray(W, dtype=float))
    if Wa.shape[0] == 1 and Wa.shape[1] != t.size:
        Wa = Wa.T
    if Wa.shape[0] != t.size:
        Wa = Wa.T
    n = t.size
    if not (ev.size == a.size == Wa.shape[0] == n):
        raise ValueError(
            "time, event, treatment and covariates must agree in length, "
            "got %d, %d, %d and %d."
            % (n, ev.size, a.size, Wa.shape[0])
        )
    if not np.all(np.isin(ev, (0.0, 1.0))):
        raise ValueError("event must be binary 0/1 (1 = failure observed).")
    if not np.all(np.isin(a, (0.0, 1.0))):
        raise ValueError("treatment must be binary 0/1.")
    if min(int(a.sum()), int((1 - a).sum())) < 5:
        raise ValueError(
            "need at least 5 subjects in each arm, got %d treated and %d "
            "control." % (int(a.sum()), int((1 - a).sum()))
        )

    k_obs, edges = discretise_times(t, n_bins)
    tmax = int(k_obs.max())
    if tmax < 2:
        raise ValueError(
            "the time grid collapsed to a single bin; supply more distinct "
            "times or a larger n_bins."
        )
    j0 = tmax if t0 is None else int(np.searchsorted(edges, float(t0),
                                                     side="left") + 1)
    j0 = int(np.clip(j0, 1, tmax))

    Xg = add_intercept(Wa)
    gbeta, separated = logit_fit(Xg, a)
    g1 = np.clip(logit_predict(Xg, gbeta), trunc, 1 - trunc)
    n_trunc = int(np.sum((logit_predict(Xg, gbeta) < trunc)
                         | (logit_predict(Xg, gbeta) > 1 - trunc)))

    lam1, lam0, lc1, lc0 = fit_hazards(Wa, a, k_obs, ev.astype(int), tmax)
    arm1 = target_arm(1.0, a, k_obs, ev.astype(int), lam1, lc1, g1, j0,
                      max_iter)
    arm0 = target_arm(0.0, a, k_obs, ev.astype(int), lam0, lc0, 1.0 - g1,
                      j0, max_iter)

    eic = arm1["eic"] - arm0["eic"]
    psi = arm1["psi"] - arm0["psi"]
    se = float(np.sqrt(np.mean(eic ** 2) / n))
    return {
        "estimate": psi,
        "se": se,
        "s1": arm1["psi"],
        "s0": arm0["psi"],
        "se_s1": float(np.sqrt(np.mean(arm1["eic"] ** 2) / n)),
        "se_s0": float(np.sqrt(np.mean(arm0["eic"] ** 2) / n)),
        "curve1": arm1["curve"],
        "curve0": arm0["curve"],
        "eic": eic,
        "eif_mean": float(np.mean(eic)),
        "eif_mean_1": float(np.mean(arm1["eic"])),
        "eif_mean_0": float(np.mean(arm0["eic"])),
        "epsilon": (arm1["epsilon"], arm0["epsilon"]),
        "iterations": (arm1["iterations"], arm0["iterations"]),
        "converged": bool(arm1["converged"] and arm0["converged"]),
        "propensity": g1,
        "n_truncated": n_trunc,
        "separated": bool(separated),
        "time_grid": edges,
        "t_index": j0,
        "horizon": float(edges[j0 - 1]),
        "tmax": tmax,
        "n": n,
        "n_events": int(ev.sum()),
        "n_censored": int((1 - ev).sum()),
    }


def cheatsheet():
    return (
        "_survtmle: discrete-time hazard TMLE -- pooled hazard fits, the "
        "Cai-van der Laan clever covariate, iterated targeting to a mean "
        "influence curve below 1/n"
    )
