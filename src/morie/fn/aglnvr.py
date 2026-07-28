# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Variance of an AlphaZero training loss stream.

Silver D et al (2018), *A general reinforcement learning algorithm that
masters chess, shogi and Go through self-play*, Science
362(6419):1140-1144. The loss is eq (1) of that paper:
:math:`\\ell = (z - v)^2 - \\pi^\\top \\log p + c\\lVert\\theta\\rVert^2`.
"""

import math

import numpy as np

from ._richresult import RichResult

__all__ = ["alphazero_loss_var", "effective_sample_size"]

_METHOD = "Autocorrelation-aware variance of a training loss stream"


def effective_sample_size(x, max_lag=None):
    r"""Effective sample size of an autocorrelated sequence.

    Uses Geyer's initial positive sequence: sum the autocorrelations in
    adjacent pairs and stop at the first non-positive pair sum, which is
    consistent for a reversible chain and, unlike truncating at a fixed
    lag, does not depend on an arbitrary cutoff.

    .. math:: n_{\text{eff}} = \frac{n}{1 + 2\sum_{k\ge1}\rho_k}

    Returns
    -------
    dict with ``ess``, ``rho``, ``tau_int``, ``n_lags_used``.
    """
    v = np.asarray(x, dtype=float).ravel()
    v = v[np.isfinite(v)]
    n = v.size
    if n < 4:
        return {"ess": float(n), "rho": np.zeros(0), "tau_int": 1.0,
                "n_lags_used": 0}
    c = v - v.mean()
    denom = float(c @ c)
    if denom <= 0:
        return {"ess": float(n), "rho": np.zeros(0), "tau_int": 1.0,
                "n_lags_used": 0}
    m = int(n // 2) if max_lag is None else min(int(max_lag), n - 1)
    # autocovariance by direct summation; n is a training run, not a
    # signal, so the FFT is not worth the wrap-around care it needs
    rho = np.empty(m + 1)
    rho[0] = 1.0
    for k in range(1, m + 1):
        rho[k] = float(c[:-k] @ c[k:]) / denom
    total = 0.0
    used = 0
    k = 1
    while k + 1 <= m:
        pair = rho[k] + rho[k + 1]
        if pair <= 0:
            break
        total += pair
        used = k + 1
        k += 2
    tau = 1.0 + 2.0 * total
    tau = max(tau, 1e-12)
    return {"ess": float(n / tau), "rho": rho, "tau_int": float(tau),
            "n_lags_used": used}


def alphazero_loss_var(losses, value_loss=None, policy_loss=None,
                       reg_loss=None, alpha=0.05):
    r"""Uncertainty of a mean training loss, with the correlation kept.

    The AlphaZero objective is a sum of three terms with different
    behaviour, so the total is reported alongside its decomposition:
    the value term :math:`(z - v)^2`, the policy cross-entropy
    :math:`-\pi^\top \log p`, and the weight penalty.

    The point of the module is the standard error. Losses recorded along
    a training run are **not** independent draws: consecutive steps
    share network weights and, in self-play, share the replay buffer
    the batches are drawn from. The usual :math:`s/\sqrt{n}` treats them
    as independent and is therefore too small -- often by a large
    factor. Reporting it makes ordinary drift look like a significant
    improvement.

    The correction is to divide by the effective sample size rather than
    the number of recorded steps. ``se_inflation`` is the ratio of the
    honest standard error to the naive one, and equals
    :math:`\sqrt{\tau_{\text{int}}}` where :math:`\tau` is the
    integrated autocorrelation time. On an independent stream it sits at
    1 and nothing changes.

    Parameters
    ----------
    losses : array-like
        Total loss at each recorded step, in order. Order matters --
        shuffling destroys exactly the structure being measured.
    value_loss, policy_loss, reg_loss : array-like, optional
        The components, if recorded separately.
    alpha : float
        Two-sided level.

    Returns
    -------
    RichResult
        ``estimate`` (mean loss), ``se``, ``se_naive``,
        ``se_inflation``, ``ess``, ``tau_int``, ``ci_lower``,
        ``ci_upper``, ``variance``, and per-component means and shares.

    References
    ----------
    Silver D et al (2018) *Science* 362(6419):1140-1144, eq (1).
    Geyer CJ (1992) *Statistical Science* 7(4):473-483.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> out = alphazero_loss_var(rng.normal(2.0, 0.5, size=500))
    >>> bool(abs(out["estimate"] - 2.0) < 0.1)
    True
    >>> bool(0.7 < out["se_inflation"] < 1.4)   # independent stream
    True
    """
    x = np.asarray(losses, dtype=float).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    if n < 1:
        raise ValueError("losses must contain at least one finite value.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1); got {alpha}.")

    mean = float(np.mean(x))
    var = float(np.var(x, ddof=1)) if n > 1 else 0.0
    se_naive = math.sqrt(var / n) if n > 1 else float("nan")
    ac = effective_sample_size(x)
    ess = max(ac["ess"], 1.0)
    se = math.sqrt(var / ess) if n > 1 else float("nan")
    infl = (se / se_naive) if (n > 1 and se_naive > 0) else float("nan")

    comps = {}
    shares = {}
    for name, arr in (("value", value_loss), ("policy", policy_loss),
                      ("regularisation", reg_loss)):
        if arr is None:
            continue
        a = np.asarray(arr, dtype=float).ravel()
        if a.size != n:
            raise ValueError(
                f"{name}_loss has length {a.size} but losses has {n}."
            )
        comps[name] = {
            "mean": float(np.mean(a)),
            "variance": float(np.var(a, ddof=1)) if n > 1 else 0.0,
            "tau_int": effective_sample_size(a)["tau_int"],
        }
        shares[name] = float(np.mean(a)) / mean if mean != 0 else float("nan")

    zc = _z(1 - alpha / 2)
    out = RichResult(
        title="AlphaZero training-loss variance",
        summary_lines=[
            ("Mean loss", mean),
            ("SE (autocorrelation-aware)", se),
            ("SE (naive, independent)", se_naive),
            ("Inflation", infl),
            ("Effective sample size", ess),
            ("Recorded steps", n),
        ],
        tables=([{
            "title": "Loss components",
            "headers": ["Component", "Mean", "Variance", "tau_int",
                        "Share of total"],
            "rows": [[k, v["mean"], v["variance"], v["tau_int"], shares[k]]
                     for k, v in comps.items()],
        }] if comps else []),
        payload={
            "estimate": mean,
            "variance": var,
            "se": se,
            "se_naive": se_naive,
            "se_inflation": infl,
            "ess": float(ess),
            "tau_int": ac["tau_int"],
            "autocorrelation": ac["rho"],
            "n_lags_used": ac["n_lags_used"],
            "ci_lower": mean - zc * se if np.isfinite(se) else float("nan"),
            "ci_upper": mean + zc * se if np.isfinite(se) else float("nan"),
            "ci_naive_lower": (mean - zc * se_naive if np.isfinite(se_naive)
                               else float("nan")),
            "ci_naive_upper": (mean + zc * se_naive if np.isfinite(se_naive)
                               else float("nan")),
            "components": comps,
            "component_shares": shares,
            "n": n,
            "method": _METHOD,
        },
        interpretation=(
            f"{n} recorded steps carry the information of about {ess:.0f} "
            "independent ones, so the honest interval is wider than the "
            f"naive one by a factor of {infl:.2f}."
            if np.isfinite(infl) else "Too few steps to assess correlation."
        ),
    )
    if np.isfinite(infl) and infl > 1.5:
        out.warnings.append(
            f"The loss stream is strongly autocorrelated (integrated time "
            f"{ac['tau_int']:.1f}). Treating the {n} recorded steps as "
            "independent would understate the standard error by a factor of "
            f"{infl:.2f}, which is enough to read ordinary drift as an "
            "improvement."
        )
    if ess < 10:
        out.warnings.append(
            f"The effective sample size is only {ess:.1f}. The mean is not "
            "well determined by this run however many steps were logged."
        )
    return out


def _z(q):
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * math.erfc(-mid / math.sqrt(2.0)) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cheatsheet():
    return (
        "aglnvr: mean and variance of an AlphaZero loss stream, with the "
        "standard error corrected for autocorrelation between training steps"
    )
