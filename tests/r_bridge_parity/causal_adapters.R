# Adapters for the `causal` group: r_matchit, r_ipw, r_aipw, r_dml, r_irm,
# r_rdrobust, r_ivreg, r_synth -> morie native R functions.
# Contract: see BRIEF.md. Nothing runs at source time except building the
# example-argument data sets (seeded).

if (!exists("ADAPTERS")) ADAPTERS <- list()

`%||%` <- function(x, y) if (is.null(x)) y else x

# --- shared helpers (base R only) -------------------------------------------

# Numeric design columns for the RHS of `f`, appended to `data` under safe
# names, so the natives (which take column names) see factors / transforms
# exactly as model.matrix expands them. Returns list(data, covariates).
.cz_rhs_columns <- function(f, data) {
  rhs <- stats::delete.response(stats::terms(f, data = data))
  plain <- attr(rhs, "term.labels")
  if (all(plain %in% names(data)) && all(vapply(data[plain], is.numeric, TRUE))) {
    return(list(data = data, covariates = plain))
  }
  mm <- stats::model.matrix(rhs, data = stats::model.frame(rhs, data = data))
  mm <- mm[, colnames(mm) != "(Intercept)", drop = FALSE]
  nm <- make.names(paste0(".cz_", colnames(mm)), unique = TRUE)
  colnames(mm) <- nm
  list(data = cbind(data, as.data.frame(mm)), covariates = nm)
}

.cz_lhs_name <- function(f) {
  v <- all.vars(f[[2]])
  if (length(v) != 1L) stop("the left-hand side must be one column (the treatment)")
  v
}

.cz_binary <- function(x) {
  if (is.factor(x)) x <- as.integer(x) - 1L
  if (is.logical(x)) x <- as.integer(x)
  as.numeric(x)
}

.cz_rel <- function(a, b) {
  a <- as.numeric(a); b <- as.numeric(b)
  max(abs(a - b) / pmax(abs(b), 1e-8))
}

# morie_caus_dml_partial_lin's fold rule (set.seed(seed); sample.int(n);
# round-robin), reproduced so DoubleML can be handed the same split.
.cz_dml_folds <- function(n, n_folds, seed) {
  old <- if (exists(".Random.seed", envir = globalenv())) get(".Random.seed", envir = globalenv()) else NULL
  on.exit(if (!is.null(old)) assign(".Random.seed", old, envir = globalenv()))
  set.seed(seed)
  perm <- sample.int(n)
  lapply(seq_len(n_folds), function(i) perm[seq(i, n, by = n_folds)])
}

.cz_ols_learner <- function(Xtr, ytr) {
  q <- qr(cbind(1, Xtr))
  b <- qr.coef(q, ytr)
  b[is.na(b)] <- 0
  function(Xn) as.numeric(cbind(1, Xn) %*% b)
}

# --- example data (seeded) --------------------------------------------------

.cz_make_obs <- function(n = 2000, seed = 101) {
  set.seed(seed)
  x1 <- rnorm(n); x2 <- rnorm(n); x3 <- runif(n)
  grp <- factor(sample(c("a", "b", "c"), n, TRUE))
  t <- rbinom(n, 1, plogis(-0.4 + 0.8 * x1 - 0.5 * x2 + 0.6 * x3 + 0.4 * (grp == "b")))
  y <- 1 + 2 * t + x1 + 0.5 * x2 - x3 + 0.3 * (grp == "c") + rnorm(n)
  data.frame(y, t, x1, x2, x3, grp)
}
.cz_obs <- .cz_make_obs()

# exact-matching data: discrete covariates only
.cz_make_disc <- function(n = 1500, seed = 202) {
  set.seed(seed)
  region <- factor(sample(c("N", "S", "E", "W"), n, TRUE))
  edu <- sample(1:3, n, TRUE)
  t <- rbinom(n, 1, plogis(-0.5 + 0.3 * edu + 0.5 * (region == "N")))
  data.frame(t, region, edu, y = rnorm(n) + t)
}
.cz_disc <- .cz_make_disc()

.cz_make_rd <- function(n = 3000, seed = 303) {
  set.seed(seed)
  x <- runif(n, -1, 1)
  y <- 1 + 2 * (x >= 0.1) + 0.8 * x - 0.6 * x^2 + rnorm(n, sd = 0.5)
  list(y = y, x = x)
}
.cz_rd <- .cz_make_rd()

.cz_make_iv <- function(n = 2000, seed = 404) {
  set.seed(seed)
  w1 <- rnorm(n); w2 <- rnorm(n); z1 <- rnorm(n); z2 <- rbinom(n, 1, 0.5)
  u <- rnorm(n)
  d <- 0.5 + 0.7 * z1 + 0.5 * z2 + 0.3 * w1 + 0.6 * u + rnorm(n)
  y <- 1 + 1.5 * d + 0.4 * w1 - 0.3 * w2 + u + rnorm(n)
  data.frame(y, d, w1, w2, z1, z2)
}
.cz_iv <- .cz_make_iv()

# --- r_matchit ---------------------------------------------------------------
# method: "nearest" (default), "exact", "subclass", "cem".

.cz_matchit_native <- function(a, M) {
  f <- a$formula; data <- as.data.frame(a$data)
  method <- a$method %||% "nearest"
  tr <- .cz_lhs_name(f)
  data[[tr]] <- .cz_binary(data[[tr]])
  if (identical(method, "exact")) {
    vars <- attr(stats::terms(f, data = data), "term.labels")
    return(M$morie_matching_exact(data, tr, vars))
  }
  r <- .cz_rhs_columns(f, data)
  switch(method,
    nearest = M$morie_matching_nearest_neighbor(r$data, tr, r$covariates,
                                                n_neighbors = a$ratio %||% 1L,
                                                caliper = a$caliper,
                                                replace = a$replace %||% FALSE),
    subclass = M$morie_matching_subclassify(r$data, tr, r$covariates,
                                            n_strata = a$subclass %||% 6L),
    cem = M$morie_matching_cem(r$data, tr, r$covariates, n_bins = NA_integer_),
    stop(sprintf("method '%s' has no native adapter (nearest, exact, subclass, cem)", method))
  )
}

ADAPTERS[["r_matchit"]] <- list(
  native = .cz_matchit_native,
  reference = function(a) {
    MatchIt::matchit(a$formula, data = a$data, method = a$method %||% "nearest")
  },
  compare = function(nat, ref) {
    if (!is.null(nat$match_pairs) && nrow(nat$match_pairs)) {
      # nearest: the control matched to each treated unit (row ids)
      mm <- ref$match.matrix
      refc <- as.numeric(mm[, 1][order(as.numeric(rownames(mm)))])
      p <- nat$match_pairs[order(as.numeric(nat$match_pairs$treated_idx)), ]
      natc <- as.numeric(p$control_idx)
      list(native = c(n_pairs = nrow(p), natc), reference = c(n_pairs = sum(!is.na(refc)), refc))
    } else if (!is.null(nat$data_with_strata)) {
      # subclass: subclass id and weight of every unit
      md <- nat$data_with_strata
      list(native = c(md$`._stratum`, md$weights),
           reference = c(as.integer(ref$subclass), ref$weights))
    } else {
      # exact / cem: matching weights of the retained units (0 = dropped)
      md <- nat$matched_data
      w <- numeric(length(ref$weights)); names(w) <- names(ref$treat)
      w[rownames(md)] <- md$weights
      list(native = c(n_matched = nrow(md), w), reference = c(n_matched = sum(ref$weights > 0), ref$weights))
    }
  },
  tol = 1e-8,
  args = list(formula = t ~ x1 + x2 + x3 + grp, data = .cz_obs, method = "nearest"),
  more_args = list(
    exact = list(formula = t ~ region + edu, data = .cz_disc, method = "exact"),
    subclass = list(formula = t ~ x1 + x2 + x3 + grp, data = .cz_obs, method = "subclass"),
    cem = list(formula = t ~ x1 + x2 + x3, data = .cz_obs, method = "cem")
  ),
  more_mismatch = list(cem = TRUE),
  note = paste("morie_matching_nearest_neighbor / _exact / _subclassify / _cem by method;",
               "nearest, exact, subclass reproduce MatchIt; cem coarsens at Sturges-many QUANTILE cutpoints",
               "where MatchIt uses equal-width Sturges bins; full/optimal/genetic not adapted")
)

# --- r_ipw -------------------------------------------------------------------

ADAPTERS[["r_ipw"]] <- list(
  native = function(a, M) {
    f <- a$formula; data <- as.data.frame(a$data)
    tr <- .cz_lhs_name(f)
    data[[tr]] <- .cz_binary(data[[tr]])
    r <- .cz_rhs_columns(f, data)
    ps <- M$morie_estimate_propensity_scores(r$data, tr, r$covariates, trim = NULL)
    d <- data.frame(t = r$data[[tr]], ps = ps)
    w <- M$morie_calculate_ipw_weights(d, "t", "ps")
    print(summary(w))
    w
  },
  reference = function(a) {
    w <- do.call(ipw::ipwpoint, list(exposure = a$formula[[2]], family = "binomial", link = "logit",
                                     denominator = {f <- a$formula; f[[2]] <- NULL; f}, data = a$data))
    w$ipw.weights
  },
  compare = function(nat, ref) list(native = nat, reference = ref),
  tol = 1e-6,
  args = list(formula = t ~ x1 + x2 + x3 + grp, data = .cz_obs),
  note = paste("morie_estimate_propensity_scores (native logit IRLS, trim=NULL) +",
               "morie_calculate_ipw_weights (unstabilised 1/P(A=a|L)); the native clips ps",
               "to [0.01, 0.99], ipwpoint does not, so they differ only for units with ps outside that range")
)

# --- r_aipw ------------------------------------------------------------------

ADAPTERS[["r_aipw"]] <- list(
  native = function(a, M) {
    w <- as.data.frame(a$w)
    names(w) <- make.names(paste0("w_", names(w) %||% seq_along(w)), unique = TRUE)
    df <- cbind(data.frame(.y = as.numeric(a$y), .a = .cz_binary(a$a)), w)
    r <- .cz_rhs_columns(stats::reformulate(names(w)), df)
    gb <- a$g.bound %||% 0.025
    M$morie_estimate_aipw(r$data, ".a", ".y", r$covariates,
                          trim = c(gb, 1 - gb), trim_type = "value",
                          outcome_fit = "pooled")
  },
  reference = function(a) {
    suppressPackageStartupMessages(requireNamespace("SuperLearner"))
    if (!"package:SuperLearner" %in% search()) suppressPackageStartupMessages(attachNamespace("SuperLearner"))
    obj <- AIPW::AIPW$new(Y = a$y, A = a$a, W = a$w, Q.SL.library = "SL.glm",
                          g.SL.library = "SL.glm", k_split = 1, verbose = FALSE)
    obj$fit(); obj$summary(); obj$result
  },
  compare = function(nat, ref) {
    list(native = c(ate = nat$ate, se = nat$se),
         reference = c(ate = ref["Mean Difference", "Estimate"], se = ref["Mean Difference", "SE"]))
  },
  tol = 1e-4,
  args = list(y = .cz_obs$y, a = .cz_obs$t, w = .cz_obs[, c("x1", "x2", "x3")]),
  note = paste("morie_estimate_aipw(outcome_fit='pooled', ps clipped at AIPW's g.bound 0.025);",
               "reference AIPW::AIPW with SL.glm for Q and g and k_split = 1 (no cross-fitting):",
               "the native has no cross-fitting, so the bridge's k_split=10 default is not reproduced")
)

# --- r_dml (PLR) -------------------------------------------------------------

ADAPTERS[["r_dml"]] <- list(
  native = function(a, M) {
    data <- as.data.frame(a$data)
    X <- stats::model.matrix(stats::reformulate(a$x), data = data)[, -1, drop = FALSE]
    r <- M$morie_caus_dml_partial_lin(as.numeric(data[[a$y]]), as.numeric(data[[a$d]]), X,
                                      n_folds = a$n_folds %||% 5L, learner = .cz_ols_learner,
                                      seed = a$seed %||% 0L)
    r[c("theta", "se", "ci", "n_folds", "first_stage_r2", "method")]
  },
  reference = function(a) {
    data <- as.data.frame(a$data)
    n <- nrow(data); K <- a$n_folds %||% 5L
    te <- .cz_dml_folds(n, K, a$seed %||% 0L)
    tr <- lapply(te, function(i) setdiff(seq_len(n), i))
    dat <- DoubleML::DoubleMLData$new(data.table::as.data.table(data[, c(a$y, a$d, a$x)]),
                                      y_col = a$y, d_cols = a$d, x_cols = a$x)
    lgr::get_logger("mlr3")$set_threshold("warn")
    obj <- DoubleML::DoubleMLPLR$new(dat, mlr3::lrn("regr.lm"), mlr3::lrn("regr.lm"), n_folds = K)
    obj$set_sample_splitting(list(list(train_ids = tr, test_ids = te)))
    obj$fit()
    obj
  },
  compare = function(nat, ref) list(native = c(theta = nat$theta, se = nat$se),
                                    reference = c(theta = unname(ref$coef), se = unname(ref$se))),
  tol = 1e-8,
  args = list(data = .cz_obs, y = "y", d = "t", x = c("x1", "x2", "x3")),
  note = paste("morie_caus_dml_partial_lin with an OLS learner; reference DoubleMLPLR with regr.lm",
               "for ml_l and ml_m on the SAME folds (handed over via set_sample_splitting);",
               "the bridge's default rpart learners have no native counterpart")
)

# --- r_irm -------------------------------------------------------------------

ADAPTERS[["r_irm"]] <- list(
  native = function(a, M) {
    data <- as.data.frame(a$data)
    X <- stats::model.matrix(stats::reformulate(a$x), data = data)[, -1, drop = FALSE]
    df <- cbind(data.frame(.y = as.numeric(data[[a$y]]), .d = .cz_binary(data[[a$d]])), X)
    M$morie_estimate_irm(df, ".d", ".y", colnames(X),
                         n_folds = a$n_folds %||% 5L, random_state = a$seed %||% 42L)
  },
  reference = function(a) {
    data <- as.data.frame(a$data)
    n <- nrow(data); K <- a$n_folds %||% 5L
    folds <- get(".morie_dml_folds", envir = asNamespace("morie"))(n, K, a$seed %||% 42L)
    te <- lapply(seq_len(K), function(k) which(folds == k))
    tr <- lapply(te, function(i) setdiff(seq_len(n), i))
    dat <- DoubleML::DoubleMLData$new(data.table::as.data.table(data[, c(a$y, a$d, a$x)]),
                                      y_col = a$y, d_cols = a$d, x_cols = a$x)
    lgr::get_logger("mlr3")$set_threshold("warn")
    obj <- DoubleML::DoubleMLIRM$new(dat, mlr3::lrn("regr.lm"), mlr3::lrn("classif.log_reg"),
                                     n_folds = K, trimming_threshold = 0.01)
    obj$set_sample_splitting(list(list(train_ids = tr, test_ids = te)))
    obj$fit()
    obj
  },
  compare = function(nat, ref) list(native = c(ate = nat$ate, se = nat$se),
                                    reference = c(ate = unname(ref$coef), se = unname(ref$se))),
  tol = 1e-3,
  args = list(data = .cz_obs, y = "y", d = "t", x = c("x1", "x2", "x3")),
  note = paste("morie_estimate_irm (cross-fit logit ps clipped [0.01,0.99] + per-arm GCV-ridge outcomes);",
               "reference DoubleMLIRM with regr.lm / classif.log_reg on morie's folds:",
               "ridge (lambda >= 1e-3 on standardised X) vs OLS and sd (n-1) vs DoubleML's n divisor")
)

# --- r_rdrobust --------------------------------------------------------------

ADAPTERS[["r_rdrobust"]] <- list(
  native = function(a, M) {
    y <- as.numeric(a$y); x <- as.numeric(a$x); cc <- a$c %||% 0
    h <- a$h; b <- a$b
    if (is.null(h) || is.null(b)) {
      bw <- M$morie_rd_mserd_bandwidth(y, x, cutoff = cc, p = a$p %||% 1)  # rdrobust's mserd
      h <- h %||% bw$h; b <- b %||% bw$b
    }
    M$morie_causrddc(y, x, cutoff = cc, h = h, b = b, p = a$p %||% 1, kernel = a$kernel %||% "triangular")[
      c("estimate", "bias_corrected", "se_conventional", "se_robust", "ci_conventional",
        "ci_robust", "pvalue_robust", "h", "b", "n_left", "n_right", "method")]
  },
  reference = function(a) rdrobust::rdrobust(a$y, a$x, c = a$c %||% 0),
  compare = function(nat, ref) {
    list(native = c(nat$estimate, nat$bias_corrected, nat$se_conventional, nat$se_robust, nat$h, nat$b),
         reference = c(ref$coef[1], ref$coef[2], ref$se[1], ref$se[3], ref$bws[1, 1], ref$bws[2, 1]))
  },
  tol = 1e-6,
  args = list(y = .cz_rd$y, x = .cz_rd$x, c = 0.1),
  note = paste("morie_rd_mserd_bandwidth (rdrobust's mserd h, b) + morie_causrddc (conventional and",
               "robust bias-corrected local-linear, nn variance); causrddc's own default bandwidth is a",
               "different plug-in, so the mserd selector is called explicitly")
)

# --- r_ivreg -----------------------------------------------------------------

.cz_iv_parts <- function(f) {
  rhs <- f[[3]]
  if (!(is.call(rhs) && identical(rhs[[1]], as.name("|"))))
    stop("formula must be y ~ exog + endog | exog + instruments")
  y <- all.vars(f[[2]])
  r1 <- all.vars(rhs[[2]]); r2 <- all.vars(rhs[[3]])
  list(y = y, endog = setdiff(r1, r2), instruments = setdiff(r2, r1), exog = intersect(r1, r2))
}

ADAPTERS[["r_ivreg"]] <- list(
  native = function(a, M) {
    p <- .cz_iv_parts(a$formula)
    M$morie_iv_tsls(as.data.frame(a$data), p$y, p$endog, p$instruments,
                    exogenous = if (length(p$exog)) p$exog else NULL,
                    robust = a$robust %||% FALSE)
  },
  reference = function(a) ivreg::ivreg(a$formula, data = a$data),
  compare = function(nat, ref) {
    cf <- stats::coef(ref); se <- sqrt(diag(stats::vcov(ref)))
    nm <- nat$variable_names
    key <- ifelse(nm %in% c("const", "(Intercept)", "Intercept"), "(Intercept)", nm)
    list(native = c(nat$coefficients, nat$std_errors), reference = c(cf[key], se[key]))
  },
  tol = 1e-6,
  args = list(formula = y ~ w1 + w2 + d | w1 + w2 + z1 + z2, data = .cz_iv),
  note = "morie_iv_tsls (native k-class, kappa=1) with robust=FALSE to match ivreg's classical SEs"
)

# --- r_synth -----------------------------------------------------------------

ADAPTERS[["r_synth"]] <- list(
  native = function(a, M) {
    x <- as.data.frame(a$data)
    tu <- a$treatment.identifier
    uv <- a$unit.variable; tv <- a$time.variable
    if (is.character(tu)) tu <- unique(x[[uv]][as.character(x[[a$unit.names.variable]]) == tu])
    ctrl <- a$controls.identifier
    if (is.character(ctrl)) ctrl <- unique(x[[uv]][as.character(x[[a$unit.names.variable]]) %in% ctrl])
    keep_t <- a$time.plot %||% sort(unique(x[[tv]]))
    x <- x[x[[uv]] %in% c(tu, ctrl) & x[[tv]] %in% keep_t, , drop = FALSE]
    ttime <- max(a$time.optimize.ssr) + 1
    M$morie_synth_control(x, a$dependent, uv, tv, treated_unit = tu,
                          treatment_time = ttime, predictors = a$predictors)
  },
  reference = function(a) {
    x <- a$data
    x[] <- lapply(x, function(v) if (is.factor(v)) as.character(v) else v)
    prep <- do.call(Synth::dataprep, c(list(foo = x), a[setdiff(names(a), "data")]))
    s <- utils::capture.output(res <- Synth::synth(prep))
    list(prep = prep, synth = res)
  },
  compare = function(nat, ref) {
    w_ref <- as.numeric(ref$synth$solution.w)
    names(w_ref) <- rownames(ref$synth$solution.w)
    y_syn_ref <- as.numeric(ref$prep$Y0plot %*% ref$synth$solution.w)
    list(native = c(nat$weights[names(w_ref)], nat$time_series$synthetic),
         reference = c(w_ref, y_syn_ref))
  },
  tol = 1e-3,
  args = local({
    e <- new.env(); utils::data("synth.data", package = "Synth", envir = e)
    list(data = e$synth.data, predictors = c("X1", "X2", "X3"), dependent = "Y",
         unit.variable = "unit.num", time.variable = "year", treatment.identifier = 7,
         controls.identifier = c(29, 2, 13, 17, 32, 38), time.predictors.prior = 1984:1989,
         time.optimize.ssr = 1984:1990, unit.names.variable = "name", time.plot = 1984:1996)
  }),
  mismatch = TRUE,
  note = paste("morie_synth_control: DIFFERENT specification from Synth::dataprep+synth -- it always uses",
               "every pre-period outcome as a predictor (Synth example uses only X1..X3 means over",
               "time.predictors.prior), averages predictors over all pre-periods (not time.predictors.prior),",
               "and optimises V by its own nested MSPE search over all pre-periods; donor weights and paths differ")
)
