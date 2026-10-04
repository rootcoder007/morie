"""morie.psymet -- Psychometric analysis (CTT, reliability, factor analysis).

Short-name API (≤6 chars, no snake_case):

- :func:`crba`   -- Cronbach's coefficient alpha
- :func:`mcdo`   -- McDonald's omega total + hierarchical
- :func:`itcor`  -- Corrected item-total correlations
- :func:`adel`   -- Alpha if item deleted
- :func:`crel`   -- Composite reliability from factor loadings
- :func:`ave`    -- Average Variance Extracted
- :func:`kmo`    -- Kaiser-Meyer-Olkin sampling adequacy
- :func:`bart`   -- Bartlett's test of sphericity
- :func:`paran`  -- Horn's parallel analysis
- :func:`splhf`  -- Spearman-Brown split-half reliability
- :func:`idisc`  -- Item discrimination index

References
----------
* Cronbach (1951). Psychometrika, 16(3), 297-334.
* McDonald (1999). Test Theory: A Unified Treatment.
* Revelle (2024). psych R package.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

# ---------------------------------------------------------------------------
# Result containers (≤6 chars)
# ---------------------------------------------------------------------------


@dataclass
class RlbRes:
    """Reliability result (Cronbach's alpha)."""

    raw: float
    std: float
    avgr: float
    k: int
    n: int
    ci_lo: float = 0.0
    ci_hi: float = 0.0

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "metric": ["raw", "std", "avgr", "k", "n", "ci_lo", "ci_hi"],
                "value": [self.raw, self.std, self.avgr, self.k, self.n, self.ci_lo, self.ci_hi],
            }
        )


@dataclass
class OmgRes:
    """Omega result (McDonald's omega)."""

    total: float
    hier: float
    alpha: float
    nf: int
    expvar: float = 0.0


@dataclass
class KmoRes:
    """KMO test result."""

    msa: float
    items: dict[str, float] = field(default_factory=dict)


@dataclass
class BrtRes:
    """Bartlett's sphericity result."""

    chisq: float
    df: int
    pval: float


# ---------------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------------


def crba(
    data: pd.DataFrame | np.ndarray,
    *,
    ci: float = 0.95,
) -> RlbRes:
    """Cronbach's coefficient alpha with Feldt CI.

    Parameters
    ----------
    data : DataFrame or ndarray
        Items as columns, respondents as rows.
    ci : float
        Confidence level (default 0.95).

    Returns
    -------
    RlbRes

    Examples
    --------
    >>> from morie.fn import _array_core as np
    >>> items = np.array([[4, 5, 3, 4], [3, 4, 3, 3], [5, 5, 4, 5], [2, 3, 2, 2], [4, 4, 4, 5], [3, 3, 2, 3]], dtype=float)
    >>> r = crba(items)
    >>> (round(r.raw, 4), r.k, r.n, round(r.ci_lo, 4) < r.raw < round(r.ci_hi, 4))
    (0.9519, 4, 6, True)
    """
    X = np.asarray(data, dtype=np.float64)
    n, k = X.shape

    if k < 2:
        return RlbRes(raw=np.nan, std=np.nan, avgr=np.nan, k=k, n=n)

    item_var = np.var(X, axis=0, ddof=1)
    total_var = np.var(X.sum(axis=1), ddof=1)
    if total_var < 1e-15:
        return RlbRes(raw=np.nan, std=np.nan, avgr=0.0, k=k, n=n)

    raw = (k / (k - 1)) * (1 - item_var.sum() / total_var)

    R = np.corrcoef(X, rowvar=False)
    mask = ~np.eye(k, dtype=bool)
    avgr = R[mask].mean()
    if np.isnan(avgr):
        avgr = 0.0
    std = (k * avgr) / (1 + (k - 1) * avgr)

    from morie.fn import _stats_core as sp

    df1 = n - 1
    df2 = (n - 1) * (k - 1)
    f_lo = sp.f.ppf(1 - (1 - ci) / 2, df1, df2)
    f_hi = sp.f.ppf((1 - ci) / 2, df1, df2)

    return RlbRes(
        raw=float(raw),
        std=float(std),
        avgr=float(avgr),
        k=k,
        n=n,
        ci_lo=float(1 - (1 - raw) * f_lo),
        ci_hi=float(1 - (1 - raw) * f_hi),
    )


def _paf(R, nf: int, min_err: float = 0.001, max_iter: int = 50):
    """Principal-axis factoring as psych::fa(fm = "pa") runs it (rmorie's .morie_paf): squared
    multiple correlations on the diagonal, iterated until the total communality moves by less
    than 0.001 (at most 50 rounds); the first factor loads positively."""
    k = R.shape[0]
    r = R.copy()
    try:
        np.fill_diagonal(r, 1.0 - 1.0 / np.diag(np.linalg.inv(R)))
    except np.linalg.LinAlgError:
        np.fill_diagonal(r, 0.5)
    comm = float(np.trace(r))
    loads = np.zeros((k, nf))
    for _ in range(max_iter):
        ev, evec = np.linalg.eigh(r)
        order = np.argsort(-ev)
        ev, evec = ev[order], evec[:, order]
        loads = evec[:, :nf] * np.sqrt(np.maximum(ev[:nf], 0))
        h2 = np.sum(loads**2, axis=1)
        np.fill_diagonal(r, h2)
        err = abs(comm - float(h2.sum()))
        comm = float(h2.sum())
        if err <= min_err:
            break
    if loads[:, 0].sum() < 0:
        loads[:, 0] = -loads[:, 0]  # the general factor loads positively
    return loads


def _mm(a, b):
    return [[sum(a[i][t] * b[t][j] for t in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def _tr(a):
    return [list(r) for r in zip(*a)]


def _inv(a):
    return np.linalg.inv(np.array(a, dtype=np.float64)).tolist()


def _inv_sqrt_psd(a):
    """(A)^(-1/2) for a symmetric positive-definite A, and the sum of sqrt eigenvalues."""
    w, V = np.linalg.eigh(np.array(a, dtype=np.float64))
    w, V = list(w.tolist()), V.tolist()
    n = len(w)
    out = [[sum(V[i][t] * V[j][t] / w[t] ** 0.5 for t in range(n)) for j in range(n)] for i in range(n)]
    return out, sum(max(x, 0.0) ** 0.5 for x in w)


def _varimax(x, eps: float = 1e-5, normalize: bool = True):
    """stats::varimax: the rotation (Kaiser-normalised unless ``normalize=False``); the SVD
    step u v' is computed as the polar factor B (B'B)^(-1/2), the same matrix."""
    k, nc = len(x), len(x[0])
    sc = [sum(v * v for v in row) ** 0.5 for row in x] if normalize else [1.0] * k
    xn = [[v / sc[i] for v in row] for i, row in enumerate(x)]
    TT = [[1.0 if i == j else 0.0 for j in range(nc)] for i in range(nc)]
    d = 0.0
    for _ in range(1000):
        z = _mm(xn, TT)
        colsq = [sum(z[i][j] ** 2 for i in range(k)) for j in range(nc)]
        inner = [[z[i][j] ** 3 - z[i][j] * colsq[j] / k for j in range(nc)] for i in range(k)]
        B = _mm(_tr(xn), inner)
        isq, dsum = _inv_sqrt_psd(_mm(_tr(B), B))
        TT = _mm(B, isq)
        dpast, d = d, dsum
        if d < dpast * (1 + eps):
            break
    z = _mm(xn, TT)
    return [[v * sc[i] for v in row] for i, row in enumerate(z)], TT


def _promax(x, m: int = 4):
    """psych::Promax: varimax without Kaiser normalisation, then the least-squares fit to the
    powered target (stats::promax normalises, which moved omega_h by up to 0.009)."""
    z, rot = _varimax(x, normalize=False)
    Q = [[v * abs(v) ** (m - 1) for v in row] for row in z]
    U = _mm(_inv(_mm(_tr(z), z)), _mm(_tr(z), Q))
    dg = _inv(_mm(_tr(U), U))
    nc = len(U)
    U = [[U[i][j] * dg[j][j] ** 0.5 for j in range(nc)] for i in range(nc)]
    return _mm(z, U), _mm(rot, U)


def _schmid_leiman_g(loads):
    """General-factor loadings by the Schmid-Leiman transformation (rmorie's
    .morie_schmid_leiman_g): promax-rotate, factor the factor correlations for one general
    factor (two factors: both load sqrt(phi_12)), project the items onto it."""
    nf = len(loads[0])
    if nf < 2:
        return [row[0] for row in loads]
    P, U = _promax(loads)
    Phi = _inv(_mm(_tr(U), U))
    s = [1.0 if sum(P[i][j] for i in range(len(P))) >= 0 else -1.0 for j in range(nf)]
    P = [[P[i][j] * s[j] for j in range(nf)] for i in range(len(P))]
    Phi = [[Phi[a][b] * s[a] * s[b] for b in range(nf)] for a in range(nf)]
    if nf == 2:
        gam = [max(Phi[0][1], 0.0) ** 0.5] * 2
    else:
        gl = _paf(np.array(Phi, dtype=np.float64), 1)
        gam = [float(v) for v in gl[:, 0].tolist()]
    return [sum(P[i][j] * gam[j] for j in range(nf)) for i in range(len(P))]


def mcdo(
    data: pd.DataFrame | np.ndarray,
    nf: int = 1,
) -> OmgRes:
    """McDonald's omega total and hierarchical.

    Loadings come from principal-axis factoring (communalities iterated on the reduced
    correlation matrix). ``hier`` is the general-factor omega of the Schmid-Leiman
    transformation: the ``nf`` factors are promax-rotated, their correlations are factored
    for one general factor (with two factors both load the square root of their
    correlation), and the items are projected onto it. With ``nf = 1`` the single factor is
    the general factor, so ``hier`` equals ``total``. Same arithmetic as
    ``rmorie::morie_psymet_omega``.

    Parameters
    ----------
    data : DataFrame or ndarray
    nf : int
        Number of factors.

    Returns
    -------
    OmgRes
    """
    X = np.asarray(data, dtype=np.float64)
    n, k = X.shape
    R = np.corrcoef(X, rowvar=False)
    if not isinstance(nf, int | np.integer) or isinstance(nf, bool) or not 1 <= nf <= k - 1:
        raise ValueError(
            f"nf must be a whole number of factors between 1 and {k - 1} (one fewer than the {k} items), not {nf!r}"
        )

    evals = np.sort(np.linalg.eigvalsh(R))[::-1]
    loads = _paf(R, nf)
    g = np.asarray(_schmid_leiman_g(loads.tolist() if hasattr(loads, "tolist") else loads), dtype=np.float64)
    # items loading negatively on the general factor are reverse-keyed: flip their signs first
    # (psych::omega's flip = TRUE), or the sums below cancel and omega collapses
    key = np.where(g < 0, -1.0, 1.0)
    if (key < 0).any():
        cols = list(data.columns) if hasattr(data, "columns") else list(range(k))
        warnings.warn(
            "mcdo: reverse-keyed item(s) scored the other way round: "
            + ", ".join(str(c) for c, kk in zip(cols, key) if kk < 0),
            stacklevel=2,
        )
        X = X * key
        R = R * np.outer(key, key)
        g = g * key
    comm = np.sum(np.asarray(loads) ** 2, axis=1)
    uniq = 1 - comm
    omg_t = 1 - uniq.sum() / R.sum()
    omg_h = g.sum() ** 2 / R.sum()

    a = crba(X)
    expvar = evals[:nf].sum() / evals.sum()

    return OmgRes(
        total=float(np.clip(omg_t, 0, 1)),
        hier=float(np.clip(omg_h, 0, 1)),
        alpha=a.raw,
        nf=nf,
        expvar=float(expvar),
    )


def itcor(data: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Corrected item-total correlations.

    Returns DataFrame: item, r_total, r_corr
    """
    X = np.asarray(data, dtype=np.float64)
    n, k = X.shape
    total = X.sum(axis=1)
    names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"i{i}" for i in range(k)]

    rows = []
    for j in range(k):
        item = X[:, j]
        rt = float(np.corrcoef(item, total)[0, 1])
        rc = float(np.corrcoef(item, total - item)[0, 1])
        rows.append({"item": names[j], "r_total": rt, "r_corr": rc})
    return pd.DataFrame(rows)


def adel(data: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Alpha if each item deleted.

    Returns DataFrame: item, adel
    """
    X = np.asarray(data, dtype=np.float64)
    k = X.shape[1]
    names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"i{i}" for i in range(k)]

    rows = []
    for j in range(k):
        a = crba(np.delete(X, j, axis=1)).raw
        rows.append({"item": names[j], "adel": a})
    return pd.DataFrame(rows)


def crel(loads: np.ndarray) -> float:
    """Composite reliability from standardized factor loadings.

    CR = (Σλ)² / ((Σλ)² + Σ(1−λ²))
    """
    lam = np.asarray(loads, dtype=np.float64)
    sl = lam.sum()
    se = (1 - lam**2).sum()
    return float(sl**2 / (sl**2 + se))


def ave(loads: np.ndarray) -> float:
    """Average Variance Extracted from factor loadings.

    AVE = mean(λ²). Values ≥ 0.5 indicate convergent validity.
    """
    return float(np.mean(np.asarray(loads, dtype=np.float64) ** 2))


# ---------------------------------------------------------------------------
# Factor analysis prerequisites
# ---------------------------------------------------------------------------


def kmo(data: pd.DataFrame | np.ndarray) -> KmoRes:
    """Kaiser-Meyer-Olkin sampling adequacy.

    MSA > 0.6 adequate, > 0.8 meritorious (Kaiser 1974).
    """
    X = np.asarray(data, dtype=np.float64)
    R = np.corrcoef(X, rowvar=False)
    k = R.shape[0]

    try:
        Ri = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        Ri = np.linalg.pinv(R)

    D = np.diag(1.0 / np.sqrt(np.diag(Ri)))
    Q = -D @ Ri @ D
    np.fill_diagonal(Q, 1.0)

    mask = ~np.eye(k, dtype=bool)
    sr2 = np.sum(R[mask] ** 2)
    sq2 = np.sum(Q[mask] ** 2)
    overall = sr2 / (sr2 + sq2)

    names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"i{i}" for i in range(k)]
    items = {}
    for j in range(k):
        r2 = np.sum(R[j, mask[j]] ** 2)
        q2 = np.sum(Q[j, mask[j]] ** 2)
        items[names[j]] = float(r2 / (r2 + q2)) if (r2 + q2) > 0 else 0.0

    return KmoRes(msa=float(overall), items=items)


def bart(data: pd.DataFrame | np.ndarray) -> BrtRes:
    """Bartlett's test of sphericity.

    H0: correlation matrix = identity. Reject (p<0.05) -> factorable.
    """
    from morie.fn import _stats_core as sp

    X = np.asarray(data, dtype=np.float64)
    n, k = X.shape
    R = np.corrcoef(X, rowvar=False)

    det_R = max(np.linalg.det(R), 1e-15)
    chisq = -(n - 1 - (2 * k + 5) / 6) * np.log(det_R)
    df = k * (k - 1) // 2
    pval = 1 - sp.chi2.cdf(chisq, df)

    return BrtRes(chisq=float(chisq), df=df, pval=float(pval))


def paran(
    data: pd.DataFrame | np.ndarray,
    nsim: int = 100,
    seed: int = 42,
) -> int:
    """Horn's parallel analysis -- suggested number of factors.

    Compares observed eigenvalues to 95th percentile of random data.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(data, dtype=np.float64)
    n, k = X.shape

    R = np.corrcoef(X, rowvar=False)
    obs = np.sort(np.linalg.eigvalsh(R))[::-1]

    sim = np.zeros((nsim, k))
    for i in range(nsim):
        Rs = np.corrcoef(rng.standard_normal((n, k)), rowvar=False)
        sim[i] = np.sort(np.linalg.eigvalsh(Rs))[::-1]

    thresh = np.percentile(sim, 95, axis=0)
    return max(int(np.sum(obs > thresh)), 1)


def splhf(
    data: pd.DataFrame | np.ndarray,
    method: str = "first_last",
) -> float:
    """Spearman-Brown split-half reliability.

    method: 'first_last' or 'odd_even'.
    """
    X = np.asarray(data, dtype=np.float64)
    k = X.shape[1]

    if method == "odd_even":
        h1 = X[:, 0::2].sum(axis=1)
        h2 = X[:, 1::2].sum(axis=1)
    else:
        mid = k // 2
        h1 = X[:, :mid].sum(axis=1)
        h2 = X[:, mid:].sum(axis=1)

    r = float(np.corrcoef(h1, h2)[0, 1])
    return 2 * r / (1 + r)


def idisc(
    data: pd.DataFrame | np.ndarray,
    pct: float = 0.27,
) -> pd.DataFrame:
    """Item discrimination index (D-statistic).

    Upper/lower groups by total score (default 27%, Kelley).
    Returns DataFrame: item, d
    """
    X = np.asarray(data, dtype=np.float64)
    n, k = X.shape
    total = X.sum(axis=1)
    cut = max(int(n * pct), 1)
    si = np.argsort(total)
    lo, hi = si[:cut], si[-cut:]

    names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"i{i}" for i in range(k)]
    rows = []
    for j in range(k):
        mx = X[:, j].max()
        pu = X[hi, j].mean() / mx if mx > 0 else 0
        pl = X[lo, j].mean() / mx if mx > 0 else 0
        rows.append({"item": names[j], "d": float(pu - pl)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Backward-compat aliases (long names -> short names)
# ---------------------------------------------------------------------------

cronbach_alpha = crba
mcdonalds_omega = mcdo
item_total_correlation = itcor
alpha_if_deleted = adel
composite_reliability = crel
average_variance_extracted = ave
kmo_test = kmo
bartlett_sphericity = bart
parallel_analysis = paran
split_half_reliability = splhf
item_discrimination = idisc

ReliabilityResult = RlbRes
OmegaResult = OmgRes
KMOResult = KmoRes
BartlettResult = BrtRes
