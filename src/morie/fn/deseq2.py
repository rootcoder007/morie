r"""Differential expression for count data: DESeq2.

Love, M. I., Huber, W., & Anders, S. (2014) "Moderated estimation of fold
change and dispersion for RNA-seq data with DESeq2", *Genome Biology* 15:550.

Counts :math:`K_{ij}` for gene :math:`i` in sample :math:`j` are modelled by
a negative binomial GLM with a log link (equations 1-2):

.. math:: K_{ij} \sim \mathrm{NB}(\mu_{ij}, \alpha_i), \quad
          \mu_{ij} = s_{ij} q_{ij}, \quad
          \log q_{ij} = \sum_r x_{jr}\beta_{ir},

so :math:`\mathrm{Var}(K_{ij}) = \mu_{ij} + \alpha_i \mu_{ij}^2`. Everything
that makes DESeq2 work rather than merely exist is in how :math:`\alpha_i` and
:math:`\beta_{ir}` are estimated, and both are empirical Bayes.

**Normalisation.** Size factors by median-of-ratios: the reference is the
geometric mean of each gene across samples, and
:math:`s_j = \mathrm{median}_{i}\, K_{ij}/K_i^R` over genes with a non-zero
reference. Gene-specific :math:`s_{ij}` may be supplied instead.

**Dispersion, in three steps.**

1. *Gene-wise*: maximise the Cox-Reid adjusted likelihood (equation 7),

   .. math:: \mathrm{CR}(\alpha; \hat\mu, K) = \ell(\alpha)
             - \tfrac{1}{2}\log\det(X^t W X),

   with :math:`w_{jj} = 1/(1/\mu_j + \alpha)`. The adjustment "corrects for
   the negative bias of dispersion estimates from using the MLEs for the
   fitted values", the count analogue of Bessel's correction.
2. *Trend*: fit :math:`\alpha_{tr}(\bar\mu) = a_1/\bar\mu + \alpha_0`
   (equation 6) by gamma-family GLM regression of the gene-wise estimates on
   the mean normalised counts, iterating with genes whose dispersion/fit
   ratio falls outside :math:`[10^{-4}, 15]` excluded.
3. *MAP*: with a log-normal prior :math:`\log\alpha_i \sim
   N(\log\alpha_{tr}(\bar\mu_i), \sigma_d^2)` (equation 5), maximise
   :math:`\mathrm{CR} + \ell_i(\alpha)` (equation 9). The prior width comes
   from subtracting the expected sampling variance of a log dispersion
   estimate from the observed spread of log residuals,

   .. math:: \sigma_d^2 = \max\{s_{lr}^2 - \psi_1((m-p)/2),\; 0.25\},

   using :math:`\mathrm{Var}(\log X) = \psi_1(f/2)` for
   :math:`X \sim \chi^2_f` and the robust
   :math:`s_{lr} = \mathrm{mad}_i(\log\alpha^{gw}_i - \log\alpha_{tr})`
   (equation 8).

A gene whose :math:`\log\alpha^{gw}_i` exceeds :math:`\log\alpha_{tr} +
2 s_{lr}` is a **dispersion outlier** and keeps its gene-wise estimate
unshrunk, because "in many cases, the reason for extraordinarily high
dispersion of a gene is that it does not obey our modeling assumptions" and
shrinking it "might lead to false positives".

**Fold changes.** A zero-centred normal prior :math:`\beta_{ir} \sim
N(0, \sigma_r^2)` on every non-intercept coefficient (equation 10) turns the
GLM fit into a ridge-penalised IRLS,
:math:`\beta \leftarrow (X^t W X + \lambda I)^{-1} X^t W z` with
:math:`\lambda_r = 1/\sigma_r^2` and
:math:`z_j = \log(\mu_j/s_j) + (K_j - \mu_j)/\mu_j`. The prior width is set
by **quantile matching**: :math:`\sigma_r` is chosen so the :math:`(1-p)`
empirical quantile of :math:`|\beta^{MLE}_r|` matches the :math:`(1-p/2)`
quantile of :math:`N(0, \sigma_r^2)`, with :math:`p = 0.05`, which is what
makes the fit robust to genes with very large LFCs. Standard errors come
from the posterior's curvature at its maximum,
:math:`\Sigma_i = (X^t W X + \lambda I)^{-1}`, and contrasts follow
equations 3-4: :math:`\beta_i^c = c^t\beta_i`,
:math:`\mathrm{SE} = \sqrt{c^t \Sigma_i c}`.

Shrinkage is on by default, as in the paper, and ``beta_prior=False`` gives
the unshrunken MLE fit -- both are reported, so the bias-variance trade the
paper describes is visible rather than assumed.

**Testing.** A Wald test on the shrunken coefficient divided by its standard
error against the standard normal, and Benjamini-Hochberg adjustment.

**What median-of-ratios assumes.** The size factors are unbiased only when
differential expression is roughly balanced between up and down. Make a fifth
of the genes four-fold higher in one group and the library composition itself
moves: the size factors absorb part of the change, every unchanged gene picks
up a fold change in the other direction, and the false positive rate goes with
it -- in this module's own anchor, from 3% of null genes below p = 0.05 to
18%, entirely removed by passing the true size factors. That is a property of
the normalisation, not of the model, so it is stated here rather than left for
the user to discover: supply ``size`` when the assumption is doubtful.

Two features of the software are deliberately **not** implemented here and
are named rather than silently omitted: independent filtering of low-count
genes before multiple-testing adjustment, and Cook's-distance outlier
replacement. Both change which p-values enter the adjustment, so a result
here is the unfiltered one. The likelihood-ratio test, the regularised
logarithm transformation, and threshold-based tests are likewise out of
scope.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["deseq2", "deseq2_differential", "differential_expression", "size_factors",
           "nb_glm_fit", "dispersion_gene_wise", "dispersion_trend",
           "cox_reid_loglik", "benjamini_hochberg", "trigamma"]


def trigamma(x):
    r""":math:`\psi_1(x)`, needed for
    :math:`\mathrm{Var}(\log \chi^2_f) = \psi_1(f/2)`.

    Recurrence up to a large argument, then the asymptotic series.
    """
    x = float(x)
    if x <= 0.0:
        raise ValueError("deseq2: trigamma needs x > 0")
    tot = 0.0
    while x < 20.0:
        tot += 1.0 / (x * x)
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    # 1/x + 1/(2x^2) + 1/(6x^3) - 1/(30x^5) + 1/(42x^7) - 1/(30x^9)
    return tot + inv * (1.0 + 0.5 * inv + inv2 * (
        1.0 / 6.0 + inv2 * (-1.0 / 30.0 + inv2 * (
            1.0 / 42.0 - inv2 / 30.0))))


def _median(v):
    s = sorted(v)
    n = len(s)
    if n == 0:
        raise ValueError("deseq2: median of an empty sequence")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _mad(v):
    m = _median(v)
    return _median([abs(t - m) for t in v]) / 0.6744897501960817


def size_factors(counts):
    r"""Median-of-ratios size factors.

    :math:`K_i^R = (\prod_j K_{ij})^{1/m}` is the gene's geometric mean and
    :math:`s_j = \mathrm{median}_{i: K_i^R \ne 0} K_{ij}/K_i^R`. Genes with
    any zero count have :math:`K_i^R = 0` and drop out, which is what makes
    the estimator robust to genes present in only some samples.
    """
    K = [[float(v) for v in row] for row in counts]
    if not K or not K[0]:
        raise ValueError("deseq2: counts must be a non-empty gene x sample "
                         "matrix")
    m = len(K[0])
    ratios = [[] for _ in range(m)]
    for row in K:
        if len(row) != m:
            raise ValueError("deseq2: ragged count matrix")
        if any(v < 0 for v in row):
            raise ValueError("deseq2: counts must be non-negative")
        if any(v <= 0 for v in row):
            continue
        gm = math.exp(sum(math.log(v) for v in row) / m)
        for j in range(m):
            ratios[j].append(row[j] / gm)
    if not ratios[0]:
        raise ValueError("deseq2: no gene has a positive count in every "
                         "sample, so median-of-ratios has no reference")
    return [_median(r) for r in ratios]


def _nb_loglik(K, mu, alpha):
    """log f_NB(K; mu, alpha) summed over samples."""
    if alpha <= 0:
        tot = 0.0
        for k, m in zip(K, mu):
            tot += k * math.log(m) - m - math.lgamma(k + 1.0)
        return tot
    r = 1.0 / alpha
    tot = 0.0
    for k, m in zip(K, mu):
        tot += (math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1.0) +
                r * math.log(r / (r + m)) + k * math.log(m / (r + m)))
    return tot


def _xtwx_logdet(X, mu, alpha):
    p = len(X[0])
    M = [[0.0] * p for _ in range(p)]
    for j, row in enumerate(X):
        w = 1.0 / (1.0 / mu[j] + alpha)
        for a in range(p):
            for b in range(p):
                M[a][b] += w * row[a] * row[b]
    sign, logdet = np.linalg.slogdet(np.asarray(M, dtype=float))
    if sign <= 0:
        return float("-inf")
    return float(logdet)


def cox_reid_loglik(alpha, K, mu, X):
    r"""Equation 7: :math:`\ell(\alpha) - \tfrac12\log\det(X^t W X)`."""
    if alpha <= 0:
        raise ValueError("deseq2: alpha must be positive")
    return _nb_loglik(K, mu, alpha) - 0.5 * _xtwx_logdet(X, mu, alpha)


def nb_glm_fit(K, X, alpha, s=None, lam=None, max_iter=100, tol=1e-8,
               beta0=None):
    r"""Negative binomial GLM by (ridge-penalised) IRLS.

    The update is the paper's own,
    :math:`\beta \leftarrow (X^t W X + \lambda I)^{-1} X^t W z` with
    :math:`z_j = \log(\mu_j / s_j) + (K_j - \mu_j)/\mu_j` and
    :math:`w_{jj} = 1/(1/\mu_j + \alpha)`. ``lam`` is the vector
    :math:`\lambda_r = 1/\sigma_r^2`; ``None`` means no prior, i.e. the MLE.

    Returns ``{"beta", "mu", "sigma", "converged", "n_iter"}`` where
    ``sigma`` is :math:`(X^t W X + \lambda I)^{-1}`, the curvature-based
    covariance of the coefficients.
    """
    m = len(K)
    p = len(X[0])
    if s is None:
        s = [1.0] * m
    lam = [0.0] * p if lam is None else [float(v) for v in lam]
    if beta0 is None:
        base = max(sum(K[j] / s[j] for j in range(m)) / m, 0.1)
        beta = [math.log(base)] + [0.0] * (p - 1)
    else:
        beta = list(beta0)
    converged = False
    it = 0
    Sig = None
    for it in range(1, int(max_iter) + 1):
        mu = []
        for j in range(m):
            eta = sum(X[j][r] * beta[r] for r in range(p))
            mu.append(max(s[j] * math.exp(min(eta, 50.0)), 1e-10))
        M = [[0.0] * p for _ in range(p)]
        v = [0.0] * p
        for j in range(m):
            w = 1.0 / (1.0 / mu[j] + alpha)
            z = math.log(mu[j] / s[j]) + (K[j] - mu[j]) / mu[j]
            for a in range(p):
                v[a] += w * X[j][a] * z
                for bb in range(p):
                    M[a][bb] += w * X[j][a] * X[j][bb]
        Mr = [[M[a][bb] + (lam[a] if a == bb else 0.0) for bb in range(p)]
              for a in range(p)]
        try:
            new = [float(t) for t in
                   np.linalg.solve(np.asarray(Mr, dtype=float),
                                   np.asarray(v, dtype=float))]
        except Exception:
            raise ValueError("deseq2: the GLM design is singular; check the "
                             "design matrix for collinear columns")
        step = max(abs(new[r] - beta[r]) for r in range(p))
        beta = new
        Sig = Mr
        if step < tol:
            converged = True
            break
    mu = []
    for j in range(m):
        eta = sum(X[j][r] * beta[r] for r in range(p))
        mu.append(max(s[j] * math.exp(min(eta, 50.0)), 1e-10))
    inv = [[float(t) for t in row] for row in
           np.linalg.inv(np.asarray(Sig, dtype=float))]
    return {"beta": beta, "mu": mu, "sigma": inv, "converged": converged,
            "n_iter": it}


def _maximise_log_alpha(obj, lo=-15.0, hi=5.0, n_grid=60, refine=60):
    """Grid then golden-section refinement on log alpha."""
    best_u, best_v = lo, obj(math.exp(lo))
    for g in range(1, n_grid + 1):
        u = lo + (hi - lo) * g / float(n_grid)
        val = obj(math.exp(u))
        if val > best_v:
            best_u, best_v = u, val
    step = (hi - lo) / n_grid
    a, b = best_u - step, best_u + step
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = obj(math.exp(c)), obj(math.exp(d))
    for _ in range(refine):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = obj(math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = obj(math.exp(d))
    return math.exp(0.5 * (a + b))


def dispersion_gene_wise(K, X, s, alpha_init=0.1):
    r"""The gene-wise estimate: maximise equation 7 over :math:`\alpha`.

    An initial GLM at a method-of-moments dispersion supplies the fitted
    values :math:`\hat\mu^0`, which are then held fixed while the Cox-Reid
    adjusted likelihood is maximised, exactly as the paper describes.
    """
    fit0 = nb_glm_fit(K, X, alpha_init, s)
    mu0 = fit0["mu"]
    return _maximise_log_alpha(
        lambda a: cox_reid_loglik(a, K, mu0, X)), mu0


def dispersion_trend(mu_bar, disp, max_iter=10, tol=1e-6):
    r"""Fit :math:`\alpha_{tr}(\bar\mu) = a_1/\bar\mu + \alpha_0`
    (equation 6).

    Gamma-family GLM regression with an identity link, "not ordinary
    least-squares regression" because "the sampling distribution of the
    gene-wise dispersion estimate around the true value can be highly
    skewed", iterated with genes whose dispersion-to-fit ratio falls outside
    :math:`[10^{-4}, 15]` excluded.
    """
    keep = [i for i in range(len(disp)) if disp[i] > 0 and mu_bar[i] > 0]
    if len(keep) < 3:
        raise ValueError("deseq2: too few genes with positive dispersion to "
                         "fit the trend")
    a1, a0 = 1.0, max(1e-8, _median([disp[i] for i in keep]))
    for _ in range(int(max_iter)):
        rows = []
        for i in keep:
            fit = a1 / mu_bar[i] + a0
            if not (1e-4 <= disp[i] / fit <= 15.0):
                continue
            rows.append(i)
        if len(rows) < 3:
            rows = keep
        # gamma GLM with identity link: weights 1/fit^2 (variance
        # proportional to the square of the mean)
        M = [[0.0, 0.0], [0.0, 0.0]]
        v = [0.0, 0.0]
        for i in rows:
            fit = max(a1 / mu_bar[i] + a0, 1e-12)
            w = 1.0 / (fit * fit)
            xrow = [1.0 / mu_bar[i], 1.0]
            for a in range(2):
                v[a] += w * xrow[a] * disp[i]
                for b in range(2):
                    M[a][b] += w * xrow[a] * xrow[b]
        try:
            new = [float(t) for t in
                   np.linalg.solve(np.asarray(M, dtype=float),
                                   np.asarray(v, dtype=float))]
        except Exception:
            break
        new = [max(new[0], 0.0), max(new[1], 1e-8)]
        delta = (new[0] - a1) ** 2 + (new[1] - a0) ** 2
        a1, a0 = new
        if delta < tol:
            break
    return {"a1": a1, "a0": a0,
            "fitted": [a1 / mu_bar[i] + a0 if mu_bar[i] > 0 else a0
                       for i in range(len(mu_bar))]}


def benjamini_hochberg(p):
    """BH step-up adjusted p-values."""
    n = len(p)
    order = sorted(range(n), key=lambda i: p[i])
    adj = [0.0] * n
    prev = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        val = min(prev, p[i] * n / float(rank))
        adj[i] = val
        prev = val
    return adj


def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_ppf(pr):
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _norm_cdf(mid) < pr:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _quantile(v, pr):
    s = sorted(v)
    if not s:
        raise ValueError("deseq2: empty quantile")
    pos = pr * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (pos - lo) * (s[hi] - s[lo])


def deseq2(counts, design, contrast=None, size=None, beta_prior=True,
           quantile_p=0.05, alpha_init=0.1, min_disp=1e-8, log2=True):
    r"""Differential expression by the DESeq2 pipeline.

    Parameters
    ----------
    counts : 2-D array-like
        Integer counts, genes in rows and samples in columns.
    design : 2-D array-like or sequence
        The design matrix :math:`X` (samples in rows), or a sequence of
        group labels for the common two-or-more-group case, in which case an
        intercept plus one indicator per non-reference level is built.
    contrast : sequence of float, optional
        :math:`c` in equations 3-4. Defaults to the last coefficient.
    size : sequence of float, optional
        Size factors. Estimated by median-of-ratios when omitted.
    beta_prior : bool
        Apply the zero-centred LFC prior of equation 10. ``False`` reports
        the MLE fit. Both estimates are returned either way.
    quantile_p : float
        :math:`p` in the quantile matching that sets :math:`\sigma_r`.
    alpha_init : float
        Dispersion used for the initial GLM that supplies
        :math:`\hat\mu^0`.
    min_disp : float
        Floor on the final dispersion.
    log2 : bool
        Report fold changes on the log2 scale, as the software does; the
        equations themselves use the natural log.

    Returns
    -------
    RichResult
        ``estimate`` / ``log_fold_change`` (shrunken when ``beta_prior``),
        ``lfc_mle``, ``lfc_se``, ``stat``, ``pvalue``, ``padj``,
        ``base_mean``, ``dispersion``, ``dispersion_gene_wise``,
        ``dispersion_fit``, ``dispersion_outlier``, ``size_factors``,
        ``sigma_d2``, ``s_lr``, ``prior_sigma``, ``trend``.

    Examples
    --------
    Two groups, three replicates each::

        res = deseq2(counts, ["A", "A", "A", "B", "B", "B"])
        res["log_fold_change"][0], res["padj"][0]

    References
    ----------
    Love, Huber & Anders (2014) *Genome Biology* 15:550, equations 1-10 and
    the Methods sections on dispersion and fold-change shrinkage.
    """
    K = [[float(v) for v in row] for row in counts]
    if not K or not K[0]:
        raise ValueError("deseq2: counts must be a non-empty matrix")
    n_genes = len(K)
    m = len(K[0])

    first = design[0] if len(design) else None
    if isinstance(first, (list, tuple)):
        X = [[float(v) for v in row] for row in design]
    else:
        levels = []
        for lab in design:
            if lab not in levels:
                levels.append(lab)
        if len(levels) < 2:
            raise ValueError("deseq2: the design has only one group, so no "
                             "coefficient can be tested")
        X = [[1.0] + [1.0 if lab == lv else 0.0 for lv in levels[1:]]
             for lab in design]
    if len(X) != m:
        raise ValueError("deseq2: the design has %d rows but the counts have "
                         "%d samples" % (len(X), m))
    p = len(X[0])
    if m <= p:
        raise ValueError("deseq2: %d samples and %d coefficients leaves no "
                         "residual degrees of freedom" % (m, p))

    s = size_factors(K) if size is None else [float(v) for v in size]
    if len(s) != m or any(v <= 0 for v in s):
        raise ValueError("deseq2: size factors must be positive, one per "
                         "sample")

    base_mean = [sum(K[i][j] / s[j] for j in range(m)) / m
                 for i in range(n_genes)]

    # ---- step 1: gene-wise dispersions (equation 7) -------------------
    gw = [0.0] * n_genes
    mu0 = [None] * n_genes
    for i in range(n_genes):
        if base_mean[i] <= 0:
            gw[i] = min_disp
            mu0[i] = [1e-10] * m
            continue
        gw[i], mu0[i] = dispersion_gene_wise(K[i], X, s, alpha_init)

    # ---- step 2: the trend (equation 6) --------------------------------
    usable = [i for i in range(n_genes) if base_mean[i] > 0 and gw[i] > 0]
    trend = dispersion_trend([base_mean[i] for i in usable],
                             [gw[i] for i in usable])
    fitted = [trend["a1"] / base_mean[i] + trend["a0"] if base_mean[i] > 0
              else trend["a0"] for i in range(n_genes)]

    # ---- step 3: prior width and MAP (equations 5, 8, 9) ---------------
    resid = [math.log(gw[i]) - math.log(fitted[i]) for i in usable]
    s_lr = _mad(resid) if len(resid) > 1 else 0.0
    sigma_d2 = max(s_lr ** 2 - trigamma((m - p) / 2.0), 0.25)
    disp = [0.0] * n_genes
    outlier = [False] * n_genes
    for i in range(n_genes):
        if base_mean[i] <= 0:
            disp[i] = max(fitted[i], min_disp)
            continue
        if math.log(gw[i]) > math.log(fitted[i]) + 2.0 * s_lr:
            outlier[i] = True
            disp[i] = max(gw[i], min_disp)
            continue
        lf = math.log(fitted[i])

        def obj(a, _lf=lf, _i=i):
            return (cox_reid_loglik(a, K[_i], mu0[_i], X) -
                    (math.log(a) - _lf) ** 2 / (2.0 * sigma_d2))
        disp[i] = max(_maximise_log_alpha(obj), min_disp)

    # ---- MLE coefficients ---------------------------------------------
    mle = []
    for i in range(n_genes):
        mle.append(nb_glm_fit(K[i], X, disp[i], s))
    c = ([0.0] * (p - 1) + [1.0]) if contrast is None else \
        [float(v) for v in contrast]
    if len(c) != p:
        raise ValueError("deseq2: the contrast must have one entry per "
                         "coefficient (%d)" % p)

    def contrast_of(fit):
        beta = fit["beta"]
        val = sum(c[r] * beta[r] for r in range(p))
        var = sum(c[a] * fit["sigma"][a][b] * c[b]
                  for a in range(p) for b in range(p))
        return val, math.sqrt(max(var, 0.0))

    # ---- LFC prior width by quantile matching (equation 10) ------------
    sigma_r = [float("inf")] * p
    for r in range(1, p):
        vals = [abs(mle[i]["beta"][r]) for i in range(n_genes)
                if base_mean[i] > 0]
        if not vals:
            sigma_r[r] = 1.0
            continue
        emp = _quantile(vals, 1.0 - quantile_p)
        theo = _norm_ppf(1.0 - quantile_p / 2.0)
        sigma_r[r] = max(emp / theo, 1e-6)
    lam = [0.0] + [1.0 / (sigma_r[r] ** 2) for r in range(1, p)]

    lfc_mle, lfc_map, se_map, se_mle = [], [], [], []
    for i in range(n_genes):
        v, sd = contrast_of(mle[i])
        lfc_mle.append(v)
        se_mle.append(sd)
        if beta_prior:
            fit = nb_glm_fit(K[i], X, disp[i], s, lam,
                             beta0=mle[i]["beta"])
        else:
            fit = mle[i]
        v2, sd2 = contrast_of(fit)
        lfc_map.append(v2)
        se_map.append(sd2)

    scale = 1.0 / math.log(2.0) if log2 else 1.0
    est = [v * scale for v in lfc_map]
    est_mle = [v * scale for v in lfc_mle]
    se = [v * scale for v in se_map]
    stat = [est[i] / se[i] if se[i] > 0 else 0.0 for i in range(n_genes)]
    pval = [2.0 * (1.0 - _norm_cdf(abs(z))) for z in stat]
    padj = benjamini_hochberg(pval)

    return RichResult(payload={
        "estimate": est,
        "log_fold_change": est,
        "lfc_mle": est_mle,
        "lfc_se": se,
        "lfc_se_mle": [v * scale for v in se_mle],
        "stat": stat,
        "pvalue": pval,
        "padj": padj,
        "base_mean": base_mean,
        "dispersion": disp,
        "dispersion_gene_wise": gw,
        "dispersion_fit": fitted,
        "dispersion_outlier": outlier,
        "size_factors": s,
        "sigma_d2": sigma_d2,
        "s_lr": s_lr,
        "prior_sigma": sigma_r,
        "trend": trend,
        "beta_prior": bool(beta_prior),
        "n_genes": n_genes,
        "n_samples": m,
        "df_residual": m - p,
        "scale": "log2" if log2 else "natural log",
        "note": "independent filtering and Cook's-distance outlier "
                "replacement are NOT applied, so padj here is over all "
                "genes",
        "method": "DESeq2 negative binomial GLM with empirical Bayes "
                  "shrinkage (Love, Huber & Anders 2014)",
    })


def cheatsheet():
    return ("deseq2: RNA-seq differential expression (Love, Huber & Anders "
            "2014). NB GLM with log link, Var = mu + alpha mu^2. Size "
            "factors by median-of-ratios. Dispersion in three steps: "
            "gene-wise by COX-REID adjusted likelihood (the adjustment is "
            "Bessel's correction for GLMs), a trend alpha_tr = a1/mu + a0 "
            "fitted by gamma GLM, then MAP under a log-normal prior whose "
            "width is s_lr^2 - trigamma((m-p)/2), floored at 0.25. Genes "
            "more than 2 s_lr above the trend are dispersion OUTLIERS and "
            "are NOT shrunk. LFCs get a zero-centred normal prior whose "
            "width is set by quantile matching, making the fit ridge IRLS; "
            "SEs come from the posterior curvature. Wald test, BH. "
            "Independent filtering and Cook's outlier replacement are not "
            "implemented.")


# compact aliases
differential_expression = deseq2
deseq2_de = deseq2

# name carried over from the generated stub this replaced
deseq2_differential = deseq2
