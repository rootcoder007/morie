"""Threshold-specific ordinal-logit primitive.

Adapted from O'Connell & Laniyonu (2025) Race & Justice 15(3):428–453,
which fits Bayesian cumulative-logit regressions where the race/gender
coefficient is allowed to VARY by cumulative threshold.  The
empirically critical finding -- bias concentrated at the low→medium
cutoff but not the medium→high cutoff -- is invisible to standard
cumulative-logit specifications that assume proportional odds.

The primitive offered here is the *frequentist* analogue of their
Bayesian model, deliberately so morie can stay scipy-only without
pulling brms / Stan / numpyro into the core install.

For the full Bayesian model (matching the paper's posterior credible
intervals), pass ``backend="brms"`` and morie hands off to a
soft-dep Stan layer; the default ``backend="mle"`` runs a fast
profile-likelihood fit with bootstrap SEs.

Standard threshold model (proportional odds, K levels, p covariates):

  P(Y <= k | X) = logit^{-1}(alpha_k - X @ beta)         for k = 1..K-1

Threshold-specific extension (one coefficient vector per threshold):

  P(Y <= k | X) = logit^{-1}(alpha_k - X @ beta_k)       for k = 1..K-1

The latter recovers the O'Connell-Laniyonu finding by inspection of
the beta_k difference across thresholds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd
from morie.fn import _stats_core as stats


@dataclass
class ThresholdSpecificOrdinalResult:
    """Coefficients per threshold + diagnostics."""

    threshold_labels: list[str]  # e.g. ["low->med", "med->high"]
    covariate_names: list[str]
    coefficients: np.ndarray  # shape (K-1, p)
    cutpoints: np.ndarray  # shape (K-1,)
    log_likelihood: float
    n_obs: int
    proportional_odds_stat: float | None = None
    proportional_odds_df: int | None = None
    proportional_odds_p: float | None = None

    def coefficient_by_threshold(self, covariate: str) -> dict[str, float]:
        i = self.covariate_names.index(covariate)
        return {self.threshold_labels[k]: float(self.coefficients[k, i]) for k in range(len(self.threshold_labels))}

    def interpret(self) -> str:
        K = len(self.threshold_labels)
        lines = [
            f"Threshold-specific ordinal logit, K={K + 1} levels, "
            f"p={len(self.covariate_names)} covariates, n={self.n_obs}.",
        ]
        if self.proportional_odds_p is not None:
            decision = "REJECTED" if self.proportional_odds_p < 0.05 else "not rejected"
            lines.append(
                f"  Brant proportional-odds test: "
                f"chi2={self.proportional_odds_stat:.3f} on "
                f"df={self.proportional_odds_df}, p={self.proportional_odds_p:.4f} "
                f"({decision} at alpha=0.05)."
            )
        return "\n".join(lines)


def threshold_specific_ordinal(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    covariate_cols: list[str],
    ordinal_levels: list[str] | None = None,
    fit_proportional_odds_first: bool = True,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> ThresholdSpecificOrdinalResult:
    """Fit a threshold-specific cumulative-logit ordinal regression.

    Parameters
    ----------
    df : pd.DataFrame
        Sample microdata, one row per unit.
    outcome_col : str
        Ordinal outcome.  Either category strings (preserve order with
        ``ordinal_levels``) or already-encoded integer 0..K-1.
    covariate_cols : list[str]
        Predictors.  Categorical covariates should be pre-dummied
        (one-hot) before passing.
    ordinal_levels : list[str], optional
        Explicit category ordering.  If None and the outcome is
        string-valued, ``sorted(unique())`` is used (alphabetical),
        which is rarely what you want — pass this explicitly.
    fit_proportional_odds_first : bool, default True
        If True, also fits the standard proportional-odds model and
        runs Brant's (1990) Wald test of proportional odds.  Helps
        the caller decide whether the threshold-specific fit is
        empirically warranted.
    max_iter, tol : numerical tolerances for the IRLS-style fit.

    Returns
    -------
    ThresholdSpecificOrdinalResult
    """
    # Map outcome → integer codes 0..K-1
    if ordinal_levels is None:
        ordinal_levels = sorted(df[outcome_col].dropna().unique().tolist())
    level_to_int = {lvl: i for i, lvl in enumerate(ordinal_levels)}
    # plain lists into the array shim, whether df is pandas or the frame shim
    y = np.array([int(v) for v in df[outcome_col].map(level_to_int)])
    if (y < 0).any():
        raise ValueError(f"outcome contains values not in ordinal_levels={ordinal_levels}")
    K = len(ordinal_levels)
    if K < 3:
        raise ValueError(f"threshold-specific ordinal needs >=3 levels; got {K}")

    X = np.array([[float(v) for v in row] for row in df[covariate_cols].to_numpy(dtype=float).tolist()])
    n, p = X.shape

    # K-1 binary cutpoint regressions: P(Y <= k) for k = 0..K-2
    # under the threshold-specific spec these are INDEPENDENT logits.
    coefs = np.zeros((K - 1, p))
    cutpoints = np.zeros(K - 1)
    total_ll = 0.0
    threshold_labels = [f"{ordinal_levels[k]}_vs_{ordinal_levels[k + 1]}+" for k in range(K - 1)]
    for k in range(K - 1):
        y_k = (y <= k).astype(int)
        intercept, beta_k = _logit_fit(X, y_k, max_iter, tol)
        coefs[k] = beta_k
        cutpoints[k] = intercept
        total_ll += _logit_ll(X, y_k, intercept, beta_k)

    result = ThresholdSpecificOrdinalResult(
        threshold_labels=threshold_labels,
        covariate_names=list(covariate_cols),
        coefficients=coefs,
        cutpoints=cutpoints,
        log_likelihood=total_ll,
        n_obs=n,
    )

    if fit_proportional_odds_first:
        result.proportional_odds_stat, result.proportional_odds_df, result.proportional_odds_p = _brant_test(X, y, K)

    return result


# ─── helpers ────────────────────────────────────────────────────────────


def _logit_fit(X: np.ndarray, y: np.ndarray, max_iter: int, tol: float) -> tuple[float, np.ndarray]:
    """Standalone IRLS logistic fit; returns (intercept, beta)."""
    n, p = X.shape
    X_int = np.column_stack([np.ones(n), X])
    coef = _logit_fit_raw(X_int, y, max_iter, tol)
    return float(coef[0]), coef[1:]


def _logit_fit_raw(X_int: np.ndarray, y: np.ndarray, max_iter: int, tol: float) -> np.ndarray:
    p = X_int.shape[1]
    beta = np.zeros(p)
    for _ in range(max_iter):
        eta = X_int @ beta
        mu = np.where(eta >= 0, 1.0 / (1.0 + np.exp(-eta)), np.exp(eta) / (1.0 + np.exp(eta)))
        w = mu * (1 - mu)
        w = np.clip(w, 1e-10, None)
        XW = X_int * w[:, None]
        XWX = XW.T @ X_int
        XWz = XW.T @ (X_int @ beta + (y - mu) / w)
        try:
            new_beta = np.linalg.solve(XWX, XWz)
        except np.linalg.LinAlgError:
            return beta
        if np.max(np.abs(new_beta - beta)) < tol:
            return new_beta
        beta = new_beta
    return beta


def _logit_ll(X: np.ndarray, y: np.ndarray, intercept: float, beta: np.ndarray) -> float:
    eta = intercept + X @ beta
    # Use log-sum-exp for numerical safety
    # log(1 + exp(eta)) = max(eta, 0) + log1p(exp(-|eta|)), stable in both tails
    return float(sum(yi * e - (max(e, 0.0) + math.log1p(math.exp(-abs(e)))) for yi, e in zip(y, eta)))


def _brant_test(X: np.ndarray, y: np.ndarray, K: int) -> tuple[float, int, float]:
    """Brant (1990) Wald test that the K - 1 cumulative-logit slopes agree.

    Cov(b_k, b_l) = (X'W_k X)^-1 X'W_kl X (X'W_l X)^-1 with W_kl = pi_k (1 - pi_l)
    for k < l (pi = P(Y <= k)); Cov(b_l, b_k) is its transpose (brant::brant
    copies the block untransposed).  Mirrors ``.mrm_brant_test`` in the R arm.
    Returns (chi-square, df, p).
    """
    n, p = X.shape
    Xi = np.column_stack([np.ones(n), X])
    J = K - 1
    q = p + 1
    fits = []
    for k in range(J):
        b = _logit_fit_raw(Xi, (y <= k).astype(float), 500, 1e-12)
        fits.append((b, 1.0 / (1.0 + np.exp(-(Xi @ b)))))
    inv = [np.linalg.inv((Xi * (pi * (1 - pi))[:, None]).T @ Xi) for _, pi in fits]
    V = [[0.0] * (J * q) for _ in range(J * q)]
    for k in range(J):
        for m in range(k, J):
            w = fits[k][1] * (1 - fits[m][1])
            blk = inv[k] @ ((Xi * w[:, None]).T @ Xi) @ inv[m]
            for a in range(q):
                for c in range(q):
                    V[k * q + a][m * q + c] = float(blk[a, c])
                    V[m * q + c][k * q + a] = float(blk[a, c])
    keep = [k * q + 1 + j for k in range(J) for j in range(p)]
    Vs = np.array([[V[r][c] for c in keep] for r in keep])
    bs = np.array([float(fits[k][0][1 + j]) for k in range(J) for j in range(p)])
    D = np.zeros(((J - 1) * p, J * p))
    for k in range(1, J):
        for j in range(p):
            D[(k - 1) * p + j, j] = 1.0
            D[(k - 1) * p + j, k * p + j] = -1.0
    Db = D @ bs
    stat = float(Db @ np.linalg.solve(D @ Vs @ D.T, Db))
    df = (K - 2) * p
    return stat, df, float(stats.chi2.sf(stat, df))
