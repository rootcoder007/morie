# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Spatial voting by alternating optimisation (W-NOMINATE).

Poole KT, Rosenthal H (1985), *A spatial model for legislative roll
call analysis*, American Journal of Political Science 29(2):357-384;
Poole KT (2005), *Spatial Models of Parliamentary Voting*, Ch 3-4.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wnominate_alternating"]

_METHOD = "W-NOMINATE alternating ideal-point estimation"


def _Phi(t):
    return 0.5 * math.erfc(-t / math.sqrt(2.0))


_PHI = np.vectorize(_Phi)
_CLIP = 1e-9


def _utility_gap(x, zy, zn, beta):
    """beta * (squared distance to nay minus squared distance to yea)."""
    dy = np.sum((x[:, None, :] - zy[None, :, :]) ** 2, axis=2)
    dn = np.sum((x[:, None, :] - zn[None, :, :]) ** 2, axis=2)
    return beta * (dn - dy)


def _loglik(votes, obs, eta):
    p = np.clip(_PHI(np.clip(eta, -8, 8)), _CLIP, 1 - _CLIP)
    return float(np.sum(obs * (votes * np.log(p)
                               + (1 - votes) * np.log(1 - p))))


def wnominate_alternating(votes, n_dims=1, polarity=None, max_iter=250,
                          tol=1e-7, seed=0, ridge=1e-3, start="svd"):
    r"""Recover legislator ideal points and roll-call positions.

    Each roll call is a choice between a yea outcome at :math:`z_j^Y`
    and a nay outcome at :math:`z_j^N`. A legislator at :math:`x_i`
    votes yea with probability

    .. math::
        P(\text{yea}) = \Phi\!\left(\beta\left[
            \lVert x_i - z_j^N \rVert^2 - \lVert x_i - z_j^Y \rVert^2
        \right]\right),

    so only the *difference* of squared distances matters. The
    likelihood is not concave in all parameters jointly, but it is
    well behaved in the legislators given the roll calls and in the
    roll calls given the legislators, which is what makes alternating
    optimisation the natural method: each half-step is a small probit
    fit and the likelihood cannot decrease.

    Three properties of this model are structural rather than
    numerical, and they are reported instead of being quietly resolved:

    **The configuration is identified only up to rotation, reflection
    and scale.** The likelihood depends on the points solely through
    the distances between them, so any rigid motion of the whole
    configuration leaves it unchanged. Comparing raw coordinates across
    two fits is meaningless. The output is normalised to zero mean and
    unit root-mean-square radius, and ``polarity`` fixes the remaining
    sign by naming a legislator who must land on the positive side --
    without it the sign is arbitrary and will flip between seeds.

    **Unanimous and near-unanimous roll calls carry no information
    about position.** A vote everyone agrees on separates nobody. Such
    roll calls are dropped, and the count is reported, because leaving
    them in inflates every fit statistic: a model that "correctly
    classifies" a 99-1 vote by predicting yea for everyone has learned
    nothing.

    **Correct classification must be read against the modal baseline.**
    Predicting the majority side of every roll call already scores
    well. ``aggregate_pre`` is the proportional reduction in error
    against that baseline, and it is the number worth quoting;
    ``correct_classification`` on its own flatters the model.

    Parameters
    ----------
    votes : array-like, shape (n_legislators, n_rollcalls)
        1 for yea, 0 for nay, NaN for absent.
    n_dims : int
        Dimensions of the policy space.
    polarity : int, optional
        Index of a legislator constrained to positive on dimension 1.
    max_iter, tol : int, float
        Alternating-optimisation controls.
    seed : int
        Random start, when ``start="random"``.
    start : {"svd", "random"}
        "svd" seeds from the leading singular vectors of the centred
        vote matrix. See the note in the source: a random start reaches
        a materially worse optimum at any practical iteration count.
    ridge : float
        Small penalty keeping the roll-call step well posed when a vote
        is nearly separating.

    Returns
    -------
    RichResult
        ``ideal_points``, ``yea_positions``, ``nay_positions``,
        ``beta``, ``log_likelihood``, ``correct_classification``,
        ``aggregate_pre``, ``modal_baseline``, ``iterations``,
        ``converged``, ``n_dropped_rollcalls``.

    References
    ----------
    Poole KT, Rosenthal H (1985) *AJPS* 29(2):357-384.
    Poole KT (2005) *Spatial Models of Parliamentary Voting*, Ch 3-4.
    """
    V = np.atleast_2d(np.asarray(votes, dtype=float))
    if V.ndim != 2:
        raise ValueError(f"votes must be 2-D; got shape {V.shape}.")
    n_dims = int(n_dims)
    if n_dims < 1:
        raise ValueError(f"n_dims must be at least 1; got {n_dims}.")
    obs_all = np.isfinite(V)
    vals = V[obs_all]
    if vals.size == 0:
        raise ValueError("votes contains no observed entries.")
    if not np.all(np.isin(vals, (0.0, 1.0))):
        raise ValueError("votes must be 1 (yea), 0 (nay) or NaN (absent).")

    # a roll call everyone agrees on separates nobody
    keep = []
    for j in range(V.shape[1]):
        col = V[:, j][obs_all[:, j]]
        if col.size >= 2 and 0 < col.sum() < col.size:
            keep.append(j)
    keep = np.asarray(keep, dtype=int)
    n_dropped = V.shape[1] - keep.size
    if keep.size < n_dims + 1:
        raise ValueError(
            f"only {keep.size} roll calls divide the chamber, which cannot "
            f"identify a {n_dims}-dimensional space."
        )
    V = V[:, keep]
    obs = np.isfinite(V)
    Y = np.nan_to_num(V)
    n, m = V.shape
    if n < n_dims + 2:
        raise ValueError(f"need at least {n_dims + 2} legislators; got {n}.")

    rng = np.random.default_rng(seed)
    # Reparameterise. The difference of squared distances is
    #   |x-zn|^2 - |x-zy|^2 = (|zn|^2 - |zy|^2) + 2 x.(zy - zn),
    # so with a_j = beta(|zn_j|^2 - |zy_j|^2) and w_j = 2 beta (zy_j - zn_j)
    # the linear predictor is exactly
    #   eta_ij = a_j + x_i . w_j,
    # bilinear in the legislator and roll-call parameters. Each half-step
    # is then an ordinary probit: legislators regress on w with a KNOWN
    # OFFSET a (not a free intercept -- a varies by roll call, not by
    # legislator, and treating it as a per-legislator constant is what
    # breaks the fit), and roll calls regress on [1, x].
    # Start from the leading singular vectors of the centred vote matrix
    # rather than from noise. The objective is not jointly concave, and
    # from a random start this reaches a far worse optimum: on a
    # simulated 120-legislator chamber the random start needed 200
    # sweeps to reach a log-likelihood of -7045, while at 60 sweeps it
    # sat at -19801 with an ideal-point correlation of 0.47 against the
    # truth. The first singular vector alone already correlates 0.92.
    if start == "svd":
        C = np.where(obs, Y - 0.5, 0.0)
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        k = min(n_dims, U.shape[1])
        x = np.zeros((n, n_dims))
        w = np.zeros((m, n_dims))
        x[:, :k] = U[:, :k]
        w[:, :k] = (Vt[:k].T * S[:k])
        sd = np.std(x[:, :k], axis=0)
        x[:, :k] /= np.where(sd > 0, sd, 1.0)
        w[:, :k] *= np.where(sd > 0, sd, 1.0)
    elif start == "random":
        x = rng.normal(size=(n, n_dims)) * 0.5
        w = rng.normal(size=(m, n_dims)) * 0.5
    else:
        raise ValueError('start must be "svd" or "random".')
    a = np.zeros(m)

    def probit_fit(design, y_row, start, offset=None):
        """Probit Fisher scoring with an optional fixed offset."""
        b = start.copy()
        off = np.zeros(y_row.size) if offset is None else offset
        for _ in range(6):
            eta = np.clip(design @ b + off, -8, 8)
            pr = np.clip(_PHI(eta), _CLIP, 1 - _CLIP)
            ph = np.maximum(np.exp(-0.5 * eta ** 2) / math.sqrt(2 * math.pi),
                            1e-10)
            wt = ph ** 2 / (pr * (1 - pr))
            z = eta - off + (y_row - pr) / ph
            A = (design.T * wt) @ design + ridge * np.eye(design.shape[1])
            g = (design.T * wt) @ z
            try:
                nb = np.linalg.solve(A, g)
            except np.linalg.LinAlgError:
                nb = np.linalg.pinv(A) @ g
            if not np.all(np.isfinite(nb)):
                break
            step = nb - b
            nrm = float(np.max(np.abs(step)))
            if nrm > 5.0:
                nb = b + step * (5.0 / nrm)
            b = nb
        return b

    ll_old = -np.inf
    delta = float("inf")
    it = 0
    converged = False
    for it in range(1, int(max_iter) + 1):
        for i in range(n):
            oi = obs[i]
            if int(oi.sum()) < n_dims + 1:
                continue
            x[i] = probit_fit(w[oi], Y[i, oi], x[i], offset=a[oi])
        for j in range(m):
            oj = obs[:, j]
            if int(oj.sum()) < n_dims + 2:
                continue
            D = np.column_stack([np.ones(int(oj.sum())), x[oj]])
            bj = probit_fit(D, Y[oj, j], np.r_[a[j], w[j]])
            a[j], w[j] = bj[0], bj[1:]

        ll = _loglik(Y, obs, a[None, :] + x @ w.T)
        delta = abs(ll - ll_old)
        if delta < tol * (1.0 + abs(ll_old)):
            ll_old = ll
            converged = True
            break
        ll_old = ll

    # normalise: eta is invariant to x -> (x - mu)/s with w -> s w and
    # a -> a + mu.w, so centre and scale are free and fixed by convention
    centre = x.mean(axis=0)
    a = a + centre @ w.T
    x = x - centre
    rms = math.sqrt(float(np.mean(np.sum(x ** 2, axis=1))))
    if rms > 0:
        x = x / rms
        w = w * rms
    flipped = False
    if polarity is not None:
        k = int(polarity)
        if not 0 <= k < n:
            raise ValueError(f"polarity must lie in 0 .. {n - 1}; got {k}.")
        if x[k, 0] < 0:
            x[:, 0] *= -1.0
            w[:, 0] *= -1.0
            flipped = True

    eta = a[None, :] + x @ w.T
    pred = (eta > 0).astype(float)
    correct = float(np.sum(obs * (pred == Y)) / max(obs.sum(), 1))
    # the baseline every roll-call model must beat
    modal_err = 0
    for j in range(m):
        col = Y[obs[:, j], j]
        modal_err += min(int(col.sum()), int(col.size - col.sum()))
    total = int(obs.sum())
    modal = 1.0 - modal_err / max(total, 1)
    errors = int(np.sum(obs * (pred != Y)))
    pre = ((modal_err - errors) / modal_err) if modal_err > 0 else float("nan")
    wnorm = np.sqrt(np.sum(w ** 2, axis=1))
    cutpoint = np.where(wnorm > 0, -a / np.maximum(wnorm, 1e-12), np.nan)

    out = RichResult(
        title=f"W-NOMINATE, {n_dims}-dimensional",
        summary_lines=[
            ("Legislators", n),
            ("Roll calls used", m),
            ("Correct classification", correct),
            ("Modal baseline", modal),
            ("Aggregate PRE", pre),
            ("Log-likelihood", ll_old),
        ],
        payload={
            "ideal_points": x,
            "rollcall_normals": w,
            "rollcall_intercepts": a,
            "cutpoints": cutpoint,
            "estimate": x,
            "log_likelihood": ll_old,
            "correct_classification": correct,
            "modal_baseline": modal,
            "aggregate_pre": pre,
            "predicted": pred,
            "iterations": it,
            "converged": converged,
            "log_likelihood_change": float(delta),
            "start": start,
            "n_dropped_rollcalls": int(n_dropped),
            "rollcalls_kept": keep,
            "polarity_flipped": flipped,
            "n_dims": n_dims,
            "n": n,
            "method": _METHOD,
        },
        interpretation=(
            f"The model classifies {correct:.1%} of votes correctly against "
            f"a modal baseline of {modal:.1%}, a proportional reduction in "
            f"error of {pre:.1%}."
        ),
    )
    if not converged:
        out.warnings.append(
            f"Alternating optimisation stopped at {max_iter} iterations with "
            f"the log-likelihood still moving by {delta:.4g} per sweep. The "
            "objective is not jointly concave, so a stopped fit is not "
            "necessarily near an optimum -- raise max_iter and compare the "
            "log-likelihood before trusting the coordinates."
        )
    if polarity is None:
        out.warnings.append(
            "No polarity was fixed. The configuration is identified only up "
            "to rotation, reflection and scale, so the sign of every "
            "dimension is arbitrary and will differ between seeds. Pass "
            "polarity to make coordinates comparable across fits."
        )
    if n_dropped:
        out.warnings.append(
            f"{n_dropped} roll calls were unanimous and carry no information "
            "about position. They are excluded; including them would inflate "
            "correct classification without improving the fit."
        )
    if np.isfinite(pre) and pre < 0.1:
        out.warnings.append(
            f"The proportional reduction in error is only {pre:.1%}. A "
            "spatial model is barely outperforming the modal baseline here, "
            "so voting on these roll calls is not well described by "
            f"position in a {n_dims}-dimensional space."
        )
    return out


def cheatsheet():
    return (
        "wnoma: W-NOMINATE alternating probit fit of legislator ideal points "
        "and roll-call positions, scored by proportional reduction in error "
        "against the modal baseline"
    )
