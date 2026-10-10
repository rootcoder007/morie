# Adapters for the `tests` group of morie R-bridge commands.
# Each native() calls morie's own R functions (statistics.R family, all returning a
# "morie_test_result" with a print method); reference() is the function SPECS names today.
if (!exists("ADAPTERS")) ADAPTERS <- list()

# ---- shared helpers (base R only) -------------------------------------------
.tst_first <- function(a, names) {           # first non-NULL of the given arg names, else a[[1]]
  for (n in names) if (!is.null(a[[n]])) return(a[[n]])
  if (length(a)) a[[1]] else NULL
}
.tst_groups <- function(formula, data) {     # y ~ g  ->  list of numeric vectors, one per level
  mf <- stats::model.frame(formula, data = data)
  if (ncol(mf) != 2L) stop("expected a formula of the form y ~ group (one grouping factor)")
  g <- mf[[2]]
  if (!is.factor(g)) g <- factor(g)
  unname(split(as.numeric(mf[[1]]), droplevels(g)))
}
.tst_alt_two_sided <- function(a, what) {
  if (!is.null(a$alternative) && a$alternative != "two.sided")
    stop(sprintf("native %s is two-sided only (alternative = '%s' not supported)", what, a$alternative))
}
.tst_conf <- function(a) if (is.null(a$conf.level)) 0.95 else a$conf.level
.tst_key <- function(r) c(statistic = r$test_statistic, p = r$p_value)
.tst_href <- function(h) c(statistic = unname(h$statistic), p = h$p.value)

# ---- simulated example data ---------------------------------------------------
local({
  set.seed(20261010)
  n <- 120
  .tst_x  <<- rnorm(n, 0.3, 1)
  .tst_y  <<- rnorm(n + 15, 0, 1.6)
  .tst_x0 <<- rnorm(n, 10, 2)
  .tst_x1 <<- .tst_x0 + rnorm(n, 0.4, 1)
  g <- factor(rep(c("a", "b", "c", "d"), times = c(40, 35, 45, 30)))
  .tst_df <<- data.frame(
    y = rnorm(length(g), mean = c(a = 0, b = 0.3, c = 0.6, d = 0.2)[as.character(g)],
              sd = c(a = 1, b = 1.4, c = 0.8, d = 1.2)[as.character(g)]),
    g = g)
  nb <- 30; k <- 4
  .tst_fr <<- data.frame(
    y = as.vector(t(outer(rnorm(nb, 0, 2), c(0, 0.3, 0.5, 0.1), "+") + rnorm(nb * k))),
    trt = factor(rep(paste0("t", 1:k), times = nb)),
    block = factor(rep(paste0("b", 1:nb), each = k)))
  .tst_cat_x <<- factor(sample(c("lo", "mid", "hi"), 300, TRUE, prob = c(.3, .4, .3)))
  .tst_cat_y <<- factor(ifelse(runif(300) < c(lo = .3, mid = .5, hi = .6)[as.character(.tst_cat_x)], "yes", "no"))
})

# ---- r_ttest(x, mu) -----------------------------------------------------------
ADAPTERS[["r_ttest"]] <- list(
  native = function(a, M) {
    .tst_alt_two_sided(a, "one_sample_ttest")
    M$one_sample_ttest(a$x, mu0 = if (is.null(a$mu)) 0 else a$mu, confidence = .tst_conf(a))
  },
  reference = function(a) stats::t.test(a$x, mu = if (is.null(a$mu)) 0 else a$mu),
  compare = function(nat, ref) list(
    native = c(.tst_key(nat), df = nat$df, lo = nat$ci_lower, hi = nat$ci_upper, est = nat$estimate),
    reference = c(.tst_href(ref), df = unname(ref$parameter), lo = ref$conf.int[1], hi = ref$conf.int[2],
                  est = unname(ref$estimate))),
  tol = 1e-10,
  args = list(x = .tst_x, mu = 0.1),
  note = "morie::one_sample_ttest (mu -> mu0); also returns Cohen's d; two-sided only"
)

# ---- r_ttest2(x, y) -----------------------------------------------------------
ADAPTERS[["r_ttest2"]] <- list(
  native = function(a, M) {
    .tst_alt_two_sided(a, "two_sample_ttest")
    M$two_sample_ttest(a$x, a$y, equal_var = isTRUE(a$var.equal), confidence = .tst_conf(a))
  },
  reference = function(a) stats::t.test(a$x, a$y),
  compare = function(nat, ref) list(
    native = c(.tst_key(nat), df = nat$df, lo = nat$ci_lower, hi = nat$ci_upper, est = nat$estimate),
    reference = c(.tst_href(ref), df = unname(ref$parameter), lo = ref$conf.int[1], hi = ref$conf.int[2],
                  est = unname(diff(rev(ref$estimate))))),
  tol = 1e-10,
  args = list(x = .tst_x, y = .tst_y),
  note = "morie::two_sample_ttest with equal_var = isTRUE(var.equal), i.e. Welch by default like t.test (native's own default is pooled, so the adapter passes it explicitly)"
)

# ---- r_paired_t(x, y) ---------------------------------------------------------
ADAPTERS[["r_paired_t"]] <- list(
  native = function(a, M) {
    .tst_alt_two_sided(a, "paired_ttest")
    M$paired_ttest(a$x, a$y, confidence = .tst_conf(a))
  },
  reference = function(a) stats::t.test(a$x, a$y, paired = TRUE),
  compare = function(nat, ref) list(
    native = c(.tst_key(nat), df = nat$df, lo = nat$ci_lower, hi = nat$ci_upper, est = nat$estimate),
    reference = c(.tst_href(ref), df = unname(ref$parameter), lo = ref$conf.int[1], hi = ref$conf.int[2],
                  est = unname(ref$estimate))),
  tol = 1e-10,
  args = list(x = .tst_x1, y = .tst_x0, paired = TRUE),
  note = "morie::paired_ttest; the bridge's fixed paired=TRUE is implied"
)

# ---- r_wilcox(x, y) -----------------------------------------------------------
ADAPTERS[["r_wilcox"]] <- list(
  native = function(a, M) {
    alt <- if (is.null(a$alternative)) "two.sided" else a$alternative
    mu <- if (is.null(a$mu)) 0 else a$mu
    if (is.null(a$y)) return(M$wilcoxon_signed_rank(a$x - mu, alternative = alt))
    if (isTRUE(a$paired)) return(M$wilcoxon_signed_rank(a$x - a$y - mu, alternative = alt))
    M$mann_whitney_u(a$x - mu, a$y, alternative = alt)
  },
  reference = function(a) stats::wilcox.test(a$x, a$y),
  compare = function(nat, ref) list(native = .tst_key(nat), reference = .tst_href(ref)),
  tol = 1e-10,
  args = list(x = .tst_x, y = .tst_y),
  note = "morie::mann_whitney_u (two samples), wilcoxon_signed_rank (y NULL or paired=TRUE); same W/V statistic, exact-vs-normal p rule as wilcox.test"
)

# ---- r_anova(formula, data) ---------------------------------------------------
ADAPTERS[["r_anova"]] <- list(
  native = function(a, M) {
    f <- .tst_first(a, c("formula", "y"))
    mf <- stats::model.frame(f, data = a$data)
    if (ncol(mf) == 2L) return(do.call(M$one_way_anova, .tst_groups(f, a$data)))
    if (ncol(mf) == 3L) {   # y ~ A + B / y ~ A * B: Type-II two-way (= aov's sequential SS when balanced)
      d <- data.frame(y = mf[[1]], A = factor(mf[[2]]), B = factor(mf[[3]]))
      names(d) <- names(mf)
      return(M$two_way_anova(d, names(mf)[1], names(mf)[2], names(mf)[3]))
    }
    stop("native ANOVA supports y ~ g (one-way) or y ~ A * B (two-way factorial) only")
  },
  reference = function(a) stats::aov(.tst_first(a, c("formula", "y")), data = a$data),
  compare = function(nat, ref) {
    s <- summary(ref)[[1]]
    list(native = c(F = nat$test_statistic, p = nat$p_value, df1 = nat$extra$df_between,
                    df2 = nat$extra$df_within),
         reference = c(F = s[["F value"]][1], p = s[["Pr(>F)"]][1], df1 = s[["Df"]][1], df2 = s[["Df"]][2]))
  },
  tol = 1e-10,
  args = list(formula = y ~ g, data = .tst_df),
  note = "morie::one_way_anova on split(y, g); two-factor formulas go to morie::two_way_anova (Type-II SS, equals aov only for balanced designs; interaction F/p in statistic, full table in $extra$anova_table); other designs unsupported"
)

# ---- r_kruskal(formula, data) -------------------------------------------------
ADAPTERS[["r_kruskal"]] <- list(
  native = function(a, M) do.call(M$kruskal_wallis, .tst_groups(.tst_first(a, c("formula", "y")), a$data)),
  reference = function(a) stats::kruskal.test(.tst_first(a, c("formula", "y")), data = a$data),
  compare = function(nat, ref) list(native = c(.tst_key(nat), df = nat$df),
                                    reference = c(.tst_href(ref), df = unname(ref$parameter))),
  tol = 1e-10,
  args = list(formula = y ~ g, data = .tst_df),
  note = "morie::kruskal_wallis on split(y, g); adds epsilon/eta^2_H effect size"
)

# ---- r_chisq(x, y) ------------------------------------------------------------
ADAPTERS[["r_chisq"]] <- list(
  native = function(a, M) {
    corr <- if (is.null(a$correct)) TRUE else a$correct
    x <- a$x
    if (is.matrix(x) || is.table(x) || is.data.frame(x)) return(M$chi2_independence(as.matrix(x), correction = corr))
    if (!is.null(a$y)) return(M$chi2_independence(table(x, a$y), correction = corr))
    M$chi2_goodness_of_fit(x, expected = a$p)
  },
  reference = function(a) stats::chisq.test(a$x, a$y),
  compare = function(nat, ref) list(native = c(.tst_key(nat), df = nat$df),
                                    reference = c(.tst_href(ref), df = unname(ref$parameter))),
  tol = 1e-10,
  args = list(x = .tst_cat_x, y = .tst_cat_y),
  note = "morie::chi2_independence (matrix, or table(x, y)), chi2_goodness_of_fit (vector alone); Yates on 2x2 as chisq.test; no simulate.p.value"
)

# ---- r_fisher(x, y) -----------------------------------------------------------
ADAPTERS[["r_fisher"]] <- list(
  native = function(a, M) {
    tab <- if (is.null(a$y)) as.matrix(a$x) else table(a$x, a$y)
    # morie's exact test is for 2 x 2 tables; a larger table goes to R's own stats::fisher.test
    if (!identical(dim(tab), c(2L, 2L))) return(stats::fisher.test(tab))
    M$fisher_exact_test(tab, alternative = if (is.null(a$alternative)) "two.sided" else a$alternative)
  },
  reference = function(a) stats::fisher.test(a$x, a$y),
  compare = function(nat, ref) list(
    native = c(p = nat$p_value, or = nat$estimate, lo = nat$ci_lower, hi = nat$ci_upper),
    reference = c(p = ref$p.value, or = unname(ref$estimate), lo = ref$conf.int[1], hi = ref$conf.int[2])),
  tol = 1e-2,
  args = list(x = factor(c(rep("exp", 40), rep("ctl", 45))),
              y = factor(c(rep(c("case", "ok"), c(22, 18)), rep(c("case", "ok"), c(12, 33))))),
  note = "morie::fisher_exact_test for 2x2 (an r x c table goes to stats::fisher.test, base R). p-value identical to 1e-16; morie solves the conditional MLE and exact CI to machine precision (matches an independent tol=1e-14 root solve exactly) while fisher.test's uniroot stops early (OR off ~1e-5..3e-4 rel, wide upper CI limits ~1e-2 rel), hence tol 1e-2"
)

# ---- r_shapiro(x) -------------------------------------------------------------
ADAPTERS[["r_shapiro"]] <- list(
  native = function(a, M) M$shapiro_wilk(a$x),
  reference = function(a) stats::shapiro.test(a$x),
  compare = function(nat, ref) list(native = .tst_key(nat), reference = .tst_href(ref)),
  tol = 1e-10,
  args = list(x = .tst_y),
  note = "morie::shapiro_wilk"
)

# ---- r_ks(x, y) ---------------------------------------------------------------
ADAPTERS[["r_ks"]] <- list(
  native = function(a, M) {
    if (is.numeric(a$y)) return(M$ks_test_two_sample(a$x, a$y))
    cdf <- if (is.null(a$y)) "pnorm" else as.character(a$y)
    extra <- a[setdiff(names(a), c("x", "y", ""))]
    M$ks_test_one_sample(a$x, cdf = cdf, args = extra)
  },
  reference = function(a) stats::ks.test(a$x, a$y),
  compare = function(nat, ref) list(native = .tst_key(nat), reference = .tst_href(ref)),
  tol = 1e-10,
  args = list(x = .tst_x, y = .tst_y),
  note = "morie::ks_test_two_sample (numeric y), ks_test_one_sample (y = CDF name like 'pnorm', extra named args passed to the CDF)"
)

# ---- r_cor(x, y, method) ------------------------------------------------------
ADAPTERS[["r_cor"]] <- list(
  native = function(a, M) {
    .tst_alt_two_sided(a, "correlation")
    method <- if (is.null(a$method)) "pearson" else match.arg(a$method, c("pearson", "spearman", "kendall"))
    switch(method,
           pearson = M$pearson_correlation(a$x, a$y, confidence = .tst_conf(a)),
           spearman = M$spearman_correlation(a$x, a$y, confidence = .tst_conf(a)),
           kendall = M$kendall_correlation(a$x, a$y))
  },
  reference = function(a) stats::cor.test(a$x, a$y, method = if (is.null(a$method)) "pearson" else a$method),
  compare = function(nat, ref) {
    n <- c(est = nat$estimate, p = nat$p_value)
    r <- c(est = unname(ref$estimate), p = ref$p.value)
    if (!is.null(ref$conf.int)) {
      n <- c(n, lo = nat$ci_lower, hi = nat$ci_upper)
      r <- c(r, lo = ref$conf.int[1], hi = ref$conf.int[2])
    }
    list(native = n, reference = r)
  },
  tol = 1e-10,
  args = list(x = .tst_x0, y = .tst_x1, method = "pearson"),
  note = "morie::pearson_correlation / spearman_correlation / kendall_correlation by method; native $test_statistic is r/rho/tau (not cor.test's t/S/T), so compare estimate+p(+CI); spearman/kendall also checked in runner"
)

# ---- r_mcnemar(x) -------------------------------------------------------------
ADAPTERS[["r_mcnemar"]] <- list(
  native = function(a, M) {
    tab <- if (is.null(a$y)) as.matrix(a$x) else table(a$x, a$y)
    if (identical(a$correct, FALSE)) stop("native mcnemar_test always applies the continuity correction (correct=TRUE)")
    M$mcnemar_test(tab)
  },
  reference = function(a) stats::mcnemar.test(a$x, a$y),
  compare = function(nat, ref) list(native = c(.tst_key(nat), df = nat$df),
                                    reference = c(.tst_href(ref), df = unname(ref$parameter))),
  tol = 1e-10,
  args = list(x = matrix(c(45, 12, 25, 38), 2, dimnames = list(before = c("+", "-"), after = c("+", "-")))),
  note = "morie::mcnemar_test (own closed-form, continuity-corrected); 2x2 only (mcnemar.test also does k x k)"
)

# ---- r_bartlett(formula, data) ------------------------------------------------
ADAPTERS[["r_bartlett"]] <- list(
  native = function(a, M) do.call(M$bartlett_test, .tst_groups(.tst_first(a, c("formula", "y")), a$data)),
  reference = function(a) stats::bartlett.test(.tst_first(a, c("formula", "y")), data = a$data),
  compare = function(nat, ref) list(native = c(.tst_key(nat), df = nat$df),
                                    reference = c(.tst_href(ref), df = unname(ref$parameter))),
  tol = 1e-10,
  args = list(formula = y ~ g, data = .tst_df),
  note = "morie::bartlett_test on split(y, g)"
)

# ---- r_levene(formula, data)  (SPECS param name for the formula is "y") ----------
ADAPTERS[["r_levene"]] <- list(
  native = function(a, M) {
    center <- if (is.null(a$center)) "median" else as.character(a$center)
    do.call(M$levene_test, c(.tst_groups(.tst_first(a, c("y", "formula")), a$data), list(center = center)))
  },
  reference = function(a) car::leveneTest(.tst_first(a, c("y", "formula")), data = a$data),
  compare = function(nat, ref) list(native = c(F = nat$test_statistic, p = nat$p_value, df1 = nat$df),
                                    reference = c(F = ref[["F value"]][1], p = ref[["Pr(>F)"]][1], df1 = ref[["Df"]][1])),
  tol = 1e-10,
  args = list(y = y ~ g, data = .tst_df),
  note = "morie::levene_test(center='median'), the Brown-Forsythe default of car::leveneTest; center='mean' also supported ('trimmed' uses 10% trim, unlike car's)"
)

# ---- r_friedman(formula, data) or r_friedman(matrix) -----------------------------
ADAPTERS[["r_friedman"]] <- list(
  native = function(a, M) {
    obj <- .tst_first(a, c("*", "formula", "y"))
    if (inherits(obj, "formula")) {
      rhs <- obj[[3]]
      if (!(is.call(rhs) && identical(rhs[[1]], as.name("|"))))
        stop("formula must be y ~ treatment | block")
      f2 <- obj
      f2[[3]] <- call("+", rhs[[2]], rhs[[3]])
      mf <- stats::model.frame(f2, data = a$data)
      trt <- factor(mf[[2]]); blk <- factor(mf[[3]])
      if (any(table(blk, trt) > 1L)) stop("not an unreplicated complete block design")
      mat <- tapply(as.numeric(mf[[1]]), list(blk, trt), function(v) v[1])
    } else {
      mat <- as.matrix(obj)
    }
    mat <- mat[stats::complete.cases(mat), , drop = FALSE]   # friedman.test drops incomplete blocks
    do.call(M$friedman_test, lapply(seq_len(ncol(mat)), function(j) mat[, j]))
  },
  reference = function(a) {
    obj <- .tst_first(a, c("*", "formula", "y"))
    if (inherits(obj, "formula")) stats::friedman.test(obj, data = a$data) else stats::friedman.test(obj)
  },
  compare = function(nat, ref) list(native = c(.tst_key(nat), df = nat$df),
                                    reference = c(.tst_href(ref), df = unname(ref$parameter))),
  tol = 1e-10,
  args = list(formula = y ~ trt | block, data = .tst_fr),
  note = "morie::friedman_test on the blocks x treatments matrix (built from y ~ trt | block, or given directly); adds Kendall's W"
)
