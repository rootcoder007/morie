r"""EMMAX: a variance component model for sample structure in GWAS.

Kang, H. M., Sul, J. H., Service, S. K., Zaitlen, N. A., Kong, S., Freimer,
N. B., Sabatti, C., & Eskin, E. (2010) "Variance component model to account
for sample structure in genome-wide association studies", *Nature Genetics*
42(4), 348-354.

Kang, H. M., Zaitlen, N. A., Wade, C. M., Kirby, A., Heckerman, D., Daly,
M. J., & Eskin, E. (2008) "Efficient control of population structure in model
organism association mapping", *Genetics* 178(3), 1709-1723. (EMMA -- the
variance component estimation EMMAX calls in its step 2, and the spectral
decomposition that makes it cheap.)

Testing one marker at a time fits :math:`y_i = \beta_0 + \beta_k X_{ik} +
\eta_{ik}` where the error absorbs every *other* marker,
:math:`\eta_{ik} = \sum_{s \ne k}\beta_s X_{is} + \epsilon_i`. That model is
misspecified if the :math:`\eta` are treated as i.i.d.: "relevant regressors
are omitted; in other words, we ignore the polygenic background of the
trait", and relatedness between individuals then shows up as inflated test
statistics. The fix is to give :math:`\eta` the covariance the polygenic
background implies.

**The procedure, in the paper's three numbered steps.**

1. Compute a pairwise relatedness matrix :math:`\hat S` (IBS, Balding-Nichols
   or any positive-semidefinite alternative) and normalise it to sample
   variance 1 by Gower centring (equation 5):

   .. math:: \hat S_N = \frac{(n-1)\hat S}{\mathrm{Tr}(P \hat S P)},
             \qquad P = I - \tfrac{1}{n}\mathbf{1}\mathbf{1}'.

2. Estimate :math:`\sigma_a^2` and :math:`\sigma_e^2` **once**, by restricted
   maximum likelihood (or ML), in

   .. math:: \mathrm{Var}(Y) = \sigma_a^2 \hat S_N + \sigma_e^2 I.
             \tag{6}

   Test :math:`H_0: \sigma_a^2 = 0`; if it is not rejected, "use ordinary
   least squares to estimate the coefficients of each of the SNPs
   genotyped". The fraction
   :math:`\sigma_a^2/(\sigma_a^2 + \sigma_e^2)` is what the paper calls
   **pseudoheritability** -- "although this is ... not directly
   interchangeable with heritability of the trait because the estimated
   pairwise relatedness does not correspond exactly to the kinship
   coefficients".

3. For each marker, a GLS F-test (or a score test) on

   .. math:: y_i = \beta_0 + \beta_k X_{ik} + \eta_i, \qquad
             \mathrm{Var}(\eta) = V \propto
             \sigma_a^2 \hat S_N + \sigma_e^2 I.
             \tag{7}

**Step 2 happening once is the whole method.** EMMAX "markedly reduces the
computational cost compared to the original EMMA by avoiding the repetitive
variance component estimation procedure for each single marker". Setting
``per_marker_reml=True`` restores the expensive exact version, so the
approximation can be measured rather than assumed -- the anchor does exactly
that.

The variance components are estimated on the spectral decomposition of
:math:`\hat S_N`, which is EMMA's contribution: with
:math:`\hat S_N = U \Lambda U'` the restricted likelihood is a
one-dimensional function of :math:`\delta = \sigma_e^2/\sigma_a^2`, so a grid
plus refinement finds the optimum without inverting a matrix per candidate.

Covariates go in as extra columns of the design, and the paper is explicit
that they belong in step 2 as well: "these additional confounding variables
should be included in the procedure of restricted maximum likelihood
estimation of the variance component parameters".

**Case-control data.** The paper adapts by regressing the 0/1 status as a
quantitative response, "in the spirit of Armitage", noting a generalised
linear mixed model would be preferable but that its "computational cost ...
is much higher". That is what ``trait="binary"`` does here; no GLMM.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["gwasem", "emmax", "emmax_gwas", "gower_normalize", "reml_variance",
           "kinship_ibs", "genomic_control"]


def _eigh(M):
    w, V = np.linalg.eigh(np.asarray(M, dtype=float))
    n = len(M)
    vals = [float(v) for v in w]
    vecs = [[float(V[i][j]) for j in range(n)] for i in range(n)]
    return vals, vecs


def kinship_ibs(genotypes):
    r"""An IBS relatedness matrix from 0/1/2 genotypes.

    :math:`\hat S_{ik} = 1 - \frac{1}{2M}\sum_j |g_{ij} - g_{kj}|`, the mean
    proportion of alleles shared identical by state, which is one of the
    matrices the paper names for step 1.
    """
    G = [[float(v) for v in row] for row in genotypes]
    n = len(G)
    if n == 0 or not G[0]:
        raise ValueError("gwasem: genotypes must be a non-empty "
                         "individual x marker matrix")
    m = len(G[0])
    if any(len(r) != m for r in G):
        raise ValueError("gwasem: ragged genotype matrix")
    S = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for k in range(i, n):
            d = sum(abs(G[i][j] - G[k][j]) for j in range(m))
            v = 1.0 - d / (2.0 * m)
            S[i][k] = S[k][i] = v
    return S


def gower_normalize(S):
    r"""Equation 5: :math:`\hat S_N = (n-1)\hat S / \mathrm{Tr}(P\hat S P)`
    with :math:`P = I - \mathbf{1}\mathbf{1}'/n`.

    Scales the relatedness matrix to sample variance 1, so
    :math:`\sigma_a^2` is on the scale of the phenotypic variance.
    """
    n = len(S)
    if n < 2:
        raise ValueError("gwasem: need at least two individuals")
    rows = [sum(r) / n for r in S]
    total = sum(rows) / n
    # Tr(PSP) = Tr(S) - 2 * mean of row means * n ... computed directly
    tr = 0.0
    for i in range(n):
        tr += S[i][i] - 2.0 * rows[i] + total
    if abs(tr) < 1e-300:
        raise ValueError("gwasem: the relatedness matrix has zero centred "
                         "trace; it carries no structure to normalise")
    f = (n - 1.0) / tr
    return [[S[i][j] * f for j in range(n)] for i in range(n)]


def _reml_delta(y, X, evals, evecs, ml=False, lo=-10.0, hi=10.0,
                n_grid=100, refine=60):
    r"""Maximise the (restricted) likelihood over
    :math:`\delta = \sigma_e^2/\sigma_a^2` on the spectral basis."""
    n = len(y)
    p = len(X[0])
    yt = [sum(evecs[i][k] * y[i] for i in range(n)) for k in range(n)]
    Xt = [[sum(evecs[i][k] * X[i][a] for i in range(n)) for a in range(p)]
          for k in range(n)]

    def loglik(delta):
        d = [evals[k] + delta for k in range(n)]
        if min(d) <= 1e-12:
            return float("-inf")
        M = [[sum(Xt[k][a] * Xt[k][b] / d[k] for k in range(n))
              for b in range(p)] for a in range(p)]
        v = [sum(Xt[k][a] * yt[k] / d[k] for k in range(n))
             for a in range(p)]
        try:
            beta = [float(t) for t in
                    np.linalg.solve(np.asarray(M, dtype=float),
                                    np.asarray(v, dtype=float))]
            sign, logdetM = np.linalg.slogdet(np.asarray(M, dtype=float))
        except Exception:
            return float("-inf")
        if sign <= 0:
            return float("-inf")
        rss = sum((yt[k] - sum(Xt[k][a] * beta[a]
                               for a in range(p))) ** 2 / d[k]
                  for k in range(n))
        if rss <= 0:
            return float("-inf")
        logdetV = sum(math.log(t) for t in d)
        if ml:
            return -0.5 * (n * math.log(2 * math.pi * rss / n) + n +
                           logdetV)
        df = n - p
        return -0.5 * (df * math.log(2 * math.pi * rss / df) + df +
                       logdetV + float(logdetM))

    best_u, best_v = lo, loglik(math.exp(lo))
    for g in range(1, n_grid + 1):
        u = lo + (hi - lo) * g / float(n_grid)
        val = loglik(math.exp(u))
        if val > best_v:
            best_u, best_v = u, val
    step = (hi - lo) / n_grid
    a, b = best_u - step, best_u + step
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = loglik(math.exp(c)), loglik(math.exp(d))
    for _ in range(refine):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = loglik(math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = loglik(math.exp(d))
    delta = math.exp(0.5 * (a + b))
    dd = [evals[k] + delta for k in range(n)]
    M = [[sum(Xt[k][a] * Xt[k][b] / dd[k] for k in range(n))
          for b in range(p)] for a in range(p)]
    v = [sum(Xt[k][a] * yt[k] / dd[k] for k in range(n)) for a in range(p)]
    beta = [float(t) for t in np.linalg.solve(np.asarray(M, dtype=float),
                                              np.asarray(v, dtype=float))]
    rss = sum((yt[k] - sum(Xt[k][a] * beta[a] for a in range(p))) ** 2 /
              dd[k] for k in range(n))
    df = n if ml else n - p
    sigma_a2 = rss / df
    return delta, sigma_a2, sigma_a2 * delta, loglik(delta)


def reml_variance(y, kinship, covariates=None, ml=False):
    r"""Step 2: estimate :math:`\sigma_a^2, \sigma_e^2` in equation 6.

    Returns ``{"sigma_a2", "sigma_e2", "delta", "pseudo_heritability",
    "loglik", "loglik_null", "lrt", "evals", "evecs"}``, where
    ``loglik_null`` is the same likelihood at :math:`\sigma_a^2 = 0` and
    ``lrt`` the statistic for :math:`H_0: \sigma_a^2 = 0`.
    """
    yv = [float(t) for t in y]
    n = len(yv)
    K = gower_normalize(kinship) if kinship is not None else None
    if K is None:
        raise ValueError("gwasem: a relatedness matrix is required")
    if len(K) != n:
        raise ValueError("gwasem: the kinship matrix must be n x n")
    if covariates is None:
        X = [[1.0] for _ in range(n)]
    else:
        X = [[1.0] + [float(v) for v in row] for row in covariates]
        if len(X) != n:
            raise ValueError("gwasem: one covariate row per individual")
    evals, evecs = _eigh(K)
    shift = -min(evals) + 1e-8 if min(evals) <= 0 else 0.0
    evals = [v + shift for v in evals]
    delta, s_a2, s_e2, ll = _reml_delta(yv, X, evals, evecs, ml)
    # null: sigma_a2 = 0, i.e. ordinary least squares
    p = len(X[0])
    M = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(p)]
         for a in range(p)]
    v = [sum(X[i][a] * yv[i] for i in range(n)) for a in range(p)]
    beta0 = [float(t) for t in np.linalg.solve(np.asarray(M, dtype=float),
                                               np.asarray(v, dtype=float))]
    rss0 = sum((yv[i] - sum(X[i][a] * beta0[a] for a in range(p))) ** 2
               for i in range(n))
    df0 = n if ml else n - p
    ll0 = -0.5 * (df0 * math.log(2 * math.pi * rss0 / df0) + df0)
    if not ml:
        sign, logdetM = np.linalg.slogdet(np.asarray(M, dtype=float))
        ll0 -= 0.5 * float(logdetM)
    return {"sigma_a2": s_a2, "sigma_e2": s_e2, "delta": delta,
            "pseudo_heritability": s_a2 / (s_a2 + s_e2)
            if s_a2 + s_e2 > 0 else 0.0,
            "loglik": ll, "loglik_null": ll0,
            "lrt": max(0.0, 2.0 * (ll - ll0)),
            "evals": evals, "evecs": evecs, "kinship_normalized": K,
            "shift": shift}


def _norm_sf(z):
    return 0.5 * math.erfc(abs(z) / math.sqrt(2.0)) * 2.0


def _f_sf(f, df1, df2):
    """Upper tail of the F distribution, via the incomplete beta."""
    if f <= 0:
        return 1.0
    x = df2 / (df2 + df1 * f)

    def betacf(a, b, x):
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        h = d
        for mm in range(1, 300):
            m2 = 2 * mm
            aa = mm * (b - mm) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
            c = 1.0 + aa / c
            if abs(c) < 1e-300:
                c = 1e-300
            h *= d * c
            aa = -(a + mm) * (qab + mm) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
            c = 1.0 + aa / c
            if abs(c) < 1e-300:
                c = 1e-300
            de = d * c
            h *= de
            if abs(de - 1.0) < 3e-16:
                break
        return h

    a, b = 0.5 * df2, 0.5 * df1
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) +
             a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta) * betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * betacf(b, a, 1.0 - x) / b


def genomic_control(stats, df=1):
    r"""The genomic control inflation factor, the median chi-square
    statistic over its null median.

    The paper reports it for every method it compares; a well-calibrated
    analysis sits at 1.
    """
    s = sorted(float(v) for v in stats)
    if not s:
        raise ValueError("gwasem: no statistics")
    n = len(s)
    med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    null_median = 0.4549364231195736 if df == 1 else float(df) * (
        1.0 - 2.0 / (9.0 * df)) ** 3
    return med / null_median


def gwasem(y, genotypes, kinship=None, covariates=None, trait="quantitative",
           test="f", ml=False, per_marker_reml=False, min_maf=0.0):
    r"""EMMAX association scan.

    Parameters
    ----------
    y : array-like
        Phenotype. With ``trait="binary"`` a 0/1 vector, regressed as a
        quantitative response "in the spirit of Armitage".
    genotypes : 2-D array-like
        Individuals in rows, markers in columns, coded as minor allele
        counts.
    kinship : 2-D array-like, optional
        :math:`\hat S`. Computed by :func:`kinship_ibs` from the genotypes
        when omitted. It is Gower-normalised internally (equation 5).
    covariates : 2-D array-like, optional
        Extra columns for the design; included in step 2 as the paper
        requires.
    test : {"f", "score"}
        The GLS F-test of step 3, or the score test the paper offers as the
        alternative.
    ml : bool
        Maximum likelihood instead of REML for the variance components.
    per_marker_reml : bool
        Re-estimate the variance components for every marker -- the
        expensive exact model EMMAX approximates. Off by default; the point
        of the method is that it is off.
    min_maf : float
        Skip markers below this minor allele frequency.

    Returns
    -------
    RichResult
        ``estimate`` / ``beta`` per marker, with ``se``, ``stat``,
        ``pvalue``; ``variance_components`` from step 2 including
        ``pseudo_heritability``; ``lambda_gc`` the genomic control inflation
        factor of the scan; ``skipped`` the markers below ``min_maf``.

    Examples
    --------
    ::

        r = gwasem(y, genotypes)
        r["variance_components"]["pseudo_heritability"], min(r["pvalue"])

    References
    ----------
    Kang et al. (2010) *Nature Genetics* 42(4), 348-354, equations 5-7 and
    the three-step Online Methods procedure; Kang et al. (2008) *Genetics*
    178, 1709-1723 for the variance component estimation of step 2.
    """
    yv = [float(t) for t in y]
    G = [[float(v) for v in row] for row in genotypes]
    n = len(yv)
    if n == 0 or len(G) != n:
        raise ValueError("gwasem: one genotype row per phenotype")
    m = len(G[0])
    if any(len(r) != m for r in G):
        raise ValueError("gwasem: ragged genotype matrix")
    if trait not in ("quantitative", "binary"):
        raise ValueError("gwasem: trait must be 'quantitative' or 'binary'")
    if trait == "binary" and any(v not in (0.0, 1.0) for v in yv):
        raise ValueError("gwasem: a binary trait must be coded 0/1")
    if test not in ("f", "score"):
        raise ValueError("gwasem: test must be 'f' or 'score'")

    K = kinship if kinship is not None else kinship_ibs(G)
    vc = reml_variance(yv, K, covariates, ml)
    evals, evecs, delta = vc["evals"], vc["evecs"], vc["delta"]

    def rotate(vec):
        return [sum(evecs[i][k] * vec[i] for i in range(n))
                for k in range(n)]

    base = [[1.0] for _ in range(n)] if covariates is None else \
        [[1.0] + [float(v) for v in row] for row in covariates]
    base_t = [rotate([row[a] for row in base])
              for a in range(len(base[0]))]
    y_t = rotate(yv)

    beta, se, stat, pval, skipped = [], [], [], [], []
    for j in range(m):
        col = [G[i][j] for i in range(n)]
        p_hat = sum(col) / (2.0 * n)
        if min(p_hat, 1 - p_hat) < min_maf or max(col) == min(col):
            skipped.append(j)
            beta.append(float("nan"))
            se.append(float("nan"))
            stat.append(0.0)
            pval.append(1.0)
            continue
        if per_marker_reml:
            Xfull = [base[i] + [col[i]] for i in range(n)]
            vcj = reml_variance(yv, K, [row[1:] + [col[i]]
                                        for i, row in enumerate(base)], ml)
            dj = vcj["delta"]
            ev = vcj["evals"]
            rot = [[sum(vcj["evecs"][i][k] * Xfull[i][a] for i in range(n))
                    for a in range(len(Xfull[0]))] for k in range(n)]
            yr = [sum(vcj["evecs"][i][k] * yv[i] for i in range(n))
                  for k in range(n)]
            d = [ev[k] + dj for k in range(n)]
        else:
            col_t = rotate(col)
            rot = [[base_t[a][k] for a in range(len(base_t))] + [col_t[k]]
                   for k in range(n)]
            yr = y_t
            d = [evals[k] + delta for k in range(n)]
        p = len(rot[0])
        M = [[sum(rot[k][a] * rot[k][b] / d[k] for k in range(n))
              for b in range(p)] for a in range(p)]
        v = [sum(rot[k][a] * yr[k] / d[k] for k in range(n))
             for a in range(p)]
        try:
            bb = [float(t) for t in
                  np.linalg.solve(np.asarray(M, dtype=float),
                                  np.asarray(v, dtype=float))]
            inv = [[float(t) for t in row] for row in
                   np.linalg.inv(np.asarray(M, dtype=float))]
        except Exception:
            skipped.append(j)
            beta.append(float("nan"))
            se.append(float("nan"))
            stat.append(0.0)
            pval.append(1.0)
            continue
        rss = sum((yr[k] - sum(rot[k][a] * bb[a] for a in range(p))) ** 2 /
                  d[k] for k in range(n))
        df = n - p
        s2 = rss / df
        b_k = bb[-1]
        var_k = s2 * inv[p - 1][p - 1]
        se_k = math.sqrt(max(var_k, 0.0))
        beta.append(b_k)
        se.append(se_k)
        if test == "f":
            f = (b_k * b_k / var_k) if var_k > 0 else 0.0
            stat.append(f)
            pval.append(_f_sf(f, 1, df))
        else:
            # Score test: everything evaluated at the NULL fit -- the
            # residuals, the residual variance and the information -- which
            # is what distinguishes it from the Wald/F statistic above.
            p0 = p - 1
            M0 = [[sum(rot[k][a] * rot[k][b] / d[k] for k in range(n))
                   for b in range(p0)] for a in range(p0)]
            v0 = [sum(rot[k][a] * yr[k] / d[k] for k in range(n))
                  for a in range(p0)]
            b0 = [float(t) for t in
                  np.linalg.solve(np.asarray(M0, dtype=float),
                                  np.asarray(v0, dtype=float))]
            r0 = [yr[k] - sum(rot[k][a] * b0[a] for a in range(p0))
                  for k in range(n)]
            s20 = sum(r0[k] ** 2 / d[k] for k in range(n)) / (n - p0)
            # residualise the marker on the null design under V^-1
            vx = [sum(rot[k][a] * rot[k][p - 1] / d[k] for k in range(n))
                  for a in range(p0)]
            cx = [float(t) for t in
                  np.linalg.solve(np.asarray(M0, dtype=float),
                                  np.asarray(vx, dtype=float))]
            xres = [rot[k][p - 1] - sum(rot[k][a] * cx[a]
                                        for a in range(p0))
                    for k in range(n)]
            num = sum(xres[k] * r0[k] / d[k] for k in range(n))
            den = sum(xres[k] * xres[k] / d[k] for k in range(n)) * s20
            chi = (num * num / den) if den > 0 else 0.0
            stat.append(chi)
            pval.append(_norm_sf(math.sqrt(max(chi, 0.0))))

    tested = [stat[j] for j in range(m) if j not in set(skipped)]
    return RichResult(payload={
        "estimate": beta,
        "beta": beta,
        "se": se,
        "stat": stat,
        "pvalue": pval,
        "variance_components": vc,
        "pseudo_heritability": vc["pseudo_heritability"],
        "lambda_gc": genomic_control(tested) if tested else float("nan"),
        "skipped": skipped,
        "n": n,
        "n_markers": m,
        "test": test,
        "trait": trait,
        "per_marker_reml": bool(per_marker_reml),
        "note": "the variance components are estimated ONCE under the null "
                "(that is what makes it EMMAX rather than EMMA); "
                "per_marker_reml=True restores the exact model",
        "method": "EMMAX variance component association (Kang et al. 2010)",
    })


def cheatsheet():
    return ("gwasem: EMMAX (Kang et al. 2010). One-marker-at-a-time "
            "regression is misspecified when relatives are present -- the "
            "omitted polygenic background inflates the statistics. Fix: "
            "Gower-normalise a relatedness matrix (eq.5), estimate "
            "sigma_a^2 and sigma_e^2 ONCE by REML in Var(Y) = sigma_a^2 "
            "S_N + sigma_e^2 I (eq.6), then GLS F-test or score test at "
            "every marker with that fixed V (eq.7). Estimating the "
            "components once instead of per marker is the eXpedited part. "
            "sigma_a^2/(sigma_a^2 + sigma_e^2) is PSEUDOheritability, not "
            "heritability. Case-control is the 0/1 response as a "
            "quantitative trait, in the spirit of Armitage; no GLMM.")


# compact aliases
emmax = gwasem
gwas_emmax = gwasem

# name carried over from the generated stub this replaced
emmax_gwas = gwasem

# public names resolved by fn/_lazy_map.json
emmaxgwas = gwasem
