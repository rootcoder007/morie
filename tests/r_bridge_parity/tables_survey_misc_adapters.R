# Adapters for GROUP tables_survey_misc: map morie R-bridge commands to morie's native R functions.
# Each entry: native(a, M) calls morie via M$...; reference(a) is the outside package the bridge
# calls today (kept only to validate the native numbers). Nothing is run at the end of this file.

if (!exists("ADAPTERS")) ADAPTERS <- list()
if (!exists("%||%")) `%||%` <- function(a, b) if (is.null(a)) b else a

.tsm <- local({
  `%||%` <- function(a, b) if (is.null(a)) b else a

  # one-sided formula / column name / NULL -> column name(s)
  fvars <- function(f) {
    if (is.null(f)) return(NULL)
    if (is.character(f)) return(f)
    v <- all.vars(f)
    if (length(v) == 0L) NULL else v
  }

  # the bridge's statistic whitelist, as base functions
  stat_names <- c("mean", "median", "sd", "var", "min", "max", "IQR", "mad", "sum")
  stat_fun <- function(name) {
    if (!is.character(name) || !name %in% stat_names)
      stop("statistic must be one of: ", paste(stat_names, collapse = ", "), call. = FALSE)
    f <- if (name %in% c("median", "sd", "var", "IQR", "mad")) getExportedValue("stats", name)
         else get(name, envir = baseenv(), mode = "function")
    function(d, i) f(d[i])
  }

  # survey design from the bridge's ids=, weights=, strata=, data=
  design <- function(a, M) {
    data <- a$data
    w <- fvars(a$weights)
    if (is.null(w)) {
      data$.w_unit <- 1
      w <- ".w_unit"
    }
    ids <- fvars(a$ids)
    if (!is.null(ids) && length(ids) > 1L)
      stop("the native design takes one stage of clusters (ids = ~psu).", call. = FALSE)
    st <- fvars(a$strata)
    M$morie_survey_design(data, weights_col = w, strata_col = st, cluster_col = ids,
                          fpc_col = fvars(a$fpc), nest = isTRUE(a$nest))
  }

  # strip "12.3 (4.5)" / "12 (33.3%)" cells into numbers
  nums <- function(s) as.numeric(regmatches(s, gregexpr("-?[0-9]+\\.?[0-9]*(e-?[0-9]+)?", s))[[1]])

  # Kaplan-Meier pieces from Surv(time, status) ~ strata
  km_parts <- function(formula, data) {
    lhs <- formula[[2]]
    if (!is.call(lhs) || !identical(as.character(lhs[[1]]), "Surv") || length(lhs) != 3L)
      stop("formula must be Surv(time, status) ~ group (or ~ 1).", call. = FALSE)
    lhs_value <- function(expr) {
      stats::model.frame(stats::as.formula(call("~", expr), env = environment(formula)), data,
                         na.action = stats::na.pass)[[1]]
    }
    time <- as.numeric(lhs_value(lhs[[2]]))
    st <- lhs_value(lhs[[3]])
    st <- if (is.logical(st)) as.numeric(st) else as.numeric(st)
    if (all(st %in% c(1, 2)) && any(st == 2)) st <- st - 1  # Surv's 1/2 coding
    rhs <- all.vars(formula[[3]])
    grp <- if (length(rhs)) interaction(data[rhs], sep = ", ", drop = TRUE) else factor(rep("all", length(time)))
    list(time = time, event = st, group = grp, rhs = rhs)
  }

  environment()
})

# Test-only helpers (example data, the reference survey design): never part of the bridge.
.tsm_test <- local({
  `%||%` <- function(a, b) if (is.null(a)) b else a

  # Build example data without disturbing the caller's RNG stream.
  sim <- function(seed, expr) {
    had <- exists(".Random.seed", envir = globalenv(), inherits = FALSE)
    old <- if (had) get(".Random.seed", envir = globalenv()) else NULL
    on.exit(if (had) assign(".Random.seed", old, envir = globalenv())
            else if (exists(".Random.seed", envir = globalenv(), inherits = FALSE))
              rm(".Random.seed", envir = globalenv()))
    set.seed(seed)
    expr
  }

  ref_design <- function(a) {
    survey::svydesign(ids = a$ids %||% ~1, weights = a$weights, strata = a$strata, data = a$data,
                      fpc = a$fpc, nest = isTRUE(a$nest))
  }

  environment()
})

# ------------------------------------------------------------------------------------------------
# r_table1(data, vars, strata): tableone::CreateTableOne  ->  morie table1()
# ------------------------------------------------------------------------------------------------
ADAPTERS[["r_table1"]] <- list(
  native = function(a, M) {
    d <- a$data
    vars <- a$vars %||% setdiff(names(d), a$strata)
    is_num <- vapply(d[vars], is.numeric, logical(1))
    M$table1(d, group_col = a$strata, continuous_vars = vars[is_num],
             categorical_vars = vars[!is_num], digits = a$digits %||% 2L)
  },
  reference = function(a) tableone::CreateTableOne(vars = a$vars, strata = a$strata, data = a$data),
  compare = function(nat, ref) {
    # native cells are formatted strings: parse means, SDs, counts, percents and p-values;
    # round the reference the same way (6 dp for mean/SD at digits = 6, 1 dp for %, 3 dp for p)
    groups <- names(ref$ContTable)
    pcol <- intersect(c("p-value", "p.value"), names(nat))[1]
    nv <- c(); rv <- c()
    cont <- rownames(attr(ref$ContTable, "pValues"))
    for (v in cont) {
      row <- grep(paste0("^", v, ", mean"), rownames(nat))
      for (g in groups) {
        nv <- c(nv, .tsm$nums(nat[row, g]))
        rv <- c(rv, round(ref$ContTable[[g]][v, c("mean", "sd")], 6))
      }
      nv <- c(nv, as.numeric(nat[row, pcol]))
      rv <- c(rv, round(attr(ref$ContTable, "pValues")[v, "pNonNormal"], 3))
    }
    cats <- rownames(attr(ref$CatTable, "pValues"))
    for (v in cats) {
      hdr <- grep(paste0("^", v, ", n"), rownames(nat))
      tab <- ref$CatTable[[groups[1]]][[v]]
      for (k in seq_along(tab$level)) {
        for (g in groups) {
          t2 <- ref$CatTable[[g]][[v]]
          nv <- c(nv, .tsm$nums(nat[hdr + k, g]))
          rv <- c(rv, t2$freq[k], round(t2$percent[k], 1))
        }
      }
      nv <- c(nv, as.numeric(nat[hdr, pcol]))
      rv <- c(rv, round(attr(ref$CatTable, "pValues")[v, "pApprox"], 3))
    }
    nv <- c(nv, as.numeric(unlist(nat["N", groups])))
    rv <- c(rv, sapply(groups, function(g) ref$ContTable[[g]][1, "n"]))
    list(native = unname(nv), reference = unname(rv))
  },
  tol = 1e-9,
  args = .tsm_test$sim(1, {
    n <- 300
    d <- data.frame(age = rnorm(n, 50, 10), bmi = rnorm(n, 27, 4),
                    sex = factor(sample(c("F", "M"), n, TRUE)),
                    stage = factor(sample(c("I", "II", "III"), n, TRUE)),
                    arm = factor(sample(c("A", "B"), n, TRUE)))
    d$age[c(3, 9)] <- NA
    list(data = d, vars = c("age", "bmi", "sex", "stage"), strata = "arm", digits = 6L)
  }),
  note = paste("table1(); means/SDs/counts/% and the categorical chi-square p agree; table1's continuous",
               "p is nonparametric (Wilcoxon/Kruskal), matched against tableone's pNonNormal (its printed",
               "default is the ANOVA pNormal); native has no 'test' column and adds SMD/Missing")
)

# ------------------------------------------------------------------------------------------------
# r_stargazer(formula, data): stargazer text table of lm  ->  morie regression_table()
# ------------------------------------------------------------------------------------------------
ADAPTERS[["r_stargazer"]] <- list(
  native = function(a, M) {
    fit <- stats::lm(a$formula, data = a$data)
    M$regression_table(list(model = fit), digits = a$digits %||% 3L, show_ci = FALSE)
  },
  reference = function(a) {
    # unqualified lm(): stargazer rejects a fit whose call reads stats::lm(...)
    fit <- do.call("lm", list(formula = a$formula, data = quote(data)), envir = list2env(list(data = a$data)))
    utils::capture.output(stargazer::stargazer(fit, type = "text"))
  },
  compare = function(nat, ref) {
    nv <- c(); rv <- c()
    terms <- nat$term[nzchar(nat$term) & !nat$term %in% c("N", "R-squared", "AIC", "BIC", "Log-Likelihood")]
    for (tm in terms) {
      i <- which(nat$term == tm)
      lab <- if (tm == "(Intercept)") "Constant" else tm
      j <- which(startsWith(trimws(ref), paste0(lab, " ")))[1]
      nv <- c(nv, .tsm$nums(nat$model[i])[1], .tsm$nums(nat$model[i + 1])[1])
      rv <- c(rv, .tsm$nums(sub(paste0("^\\s*", lab), "", ref[j]))[1], .tsm$nums(ref[j + 1])[1])
    }
    obs <- .tsm$nums(sub("Observations", "", grep("^Observations", ref, value = TRUE)))[1]
    r2 <- .tsm$nums(sub("^R2", "", grep("^R2 ", ref, value = TRUE)))[1]
    nv <- c(nv, as.numeric(nat$model[nat$term == "N"]), as.numeric(nat$model[nat$term == "R-squared"]))
    rv <- c(rv, obs, r2)
    list(native = nv, reference = rv)
  },
  tol = 1e-9,
  args = .tsm_test$sim(2, {
    n <- 300
    d <- data.frame(bmi = rnorm(n, 27, 4), sex = factor(sample(c("F", "M"), n, TRUE)))
    d$age <- 30 + 0.8 * d$bmi + 2 * (d$sex == "M") + rnorm(n, 0, 8)
    list(formula = age ~ bmi + sex, data = d)
  }),
  note = paste("regression_table() on the same lm; coefficients, SEs, N and R2 agree at 3 dp;",
               "stars differ by design (morie * p<.05/** .01/*** .001, stargazer * p<.1/** .05/*** .01);",
               "native adds AIC/BIC/logLik, lacks adjusted R2/residual SE/F")
)

# ------------------------------------------------------------------------------------------------
# r_mice(data, m, method): mice::mice  ->  morie_miord2 (MICE, Bayesian normal "norm" model)
# ------------------------------------------------------------------------------------------------
ADAPTERS[["r_mice"]] <- list(
  native = function(a, M) {
    d <- a$data
    meth <- unique(a$method %||% "norm")
    meth <- meth[nzchar(meth)]
    if (length(meth) && !all(meth == "norm"))
      stop("the native MICE (morie_miord2) imputes with method = 'norm' (Bayesian linear ",
           "regression) only; got ", paste(meth, collapse = ", "), call. = FALSE)
    if (!all(vapply(d, is.numeric, logical(1))))
      stop("the native MICE imputes numeric columns only.", call. = FALSE)
    r <- M$morie_miord2(as.matrix(d), m = a$m %||% 5L, maxit = a$maxit %||% 5L, seed = a$seed %||% 0)
    imps <- lapply(r$imputations, function(x) { x <- as.data.frame(x); names(x) <- names(d); x })
    miss <- colSums(r$missing_mask)
    out <- data.frame(variable = names(d), n_missing = miss,
                      pooled_mean = colMeans(do.call(rbind, lapply(imps, colMeans))),
                      between_imp_sd = apply(do.call(rbind, lapply(imps, colMeans)), 2, stats::sd),
                      row.names = NULL)
    attr(out, "imputations") <- imps
    attr(out, "method") <- r$method
    out
  },
  reference = function(a) {
    mice::mice(a$data, m = a$m %||% 5L, method = a$method %||% "norm", printFlag = FALSE,
               maxit = a$maxit %||% 5L)
  },
  compare = function(nat, ref) {
    comp <- lapply(seq_len(ref$m), function(i) mice::complete(ref, i))
    rm <- colMeans(do.call(rbind, lapply(comp, colMeans)))
    imps <- attr(nat, "imputations")
    # also the pooled slope of the first column on the others (Rubin point estimate)
    f <- stats::as.formula(paste(names(rm)[1], "~ ."))
    # slopes only: the intercept's Monte Carlo spread is large relative to its size
    nb <- rowMeans(sapply(imps, function(x) stats::coef(stats::lm(f, x))))[-1]
    rb <- rowMeans(sapply(comp, function(x) stats::coef(stats::lm(f, x))))[-1]
    list(native = c(nat$pooled_mean, nb), reference = c(unname(rm), unname(rb)))
  },
  tol = 0.03,
  args = .tsm_test$sim(3, {
    n <- 2000
    x1 <- rnorm(n, 10, 2); x2 <- 0.5 * x1 + rnorm(n, 0, 1.5); y <- 2 + 0.7 * x1 - 0.4 * x2 + rnorm(n)
    d <- data.frame(y = y, x1 = x1, x2 = x2)
    # one variable missing per incomplete row (Amelia leaves all-missing rows unimputed)
    i <- sample(n, 600); d$x1[i[1:200]] <- NA; d$x2[i[201:450]] <- NA; d$y[i[451:600]] <- NA
    list(data = d, m = 50L, method = "norm")
  }),
  note = paste("morie_miord2 = mice method 'norm' (van Buuren Algs 3.1+4.3) on its own SplitMix64 stream,",
               "so draws differ from mice: pooled means/coefficients agree within Monte Carlo error",
               "(tol 3%); only method='norm' and numeric columns are supported (mice's default is 'pmm';",
               "morie's mi_pmm is single-variable, no parameter draw)")
)

# ------------------------------------------------------------------------------------------------
# r_amelia(data, m): Amelia::amelia (EM with bootstrap)  ->  nearest native: morie_miord2 + em_imputation
# ------------------------------------------------------------------------------------------------
ADAPTERS[["r_amelia"]] <- list(
  native = function(a, M) {
    d <- a$x %||% a$data
    if (!all(vapply(d, is.numeric, logical(1))))
      stop("the native multivariate-normal imputation takes numeric columns only.", call. = FALSE)
    r <- M$morie_miord2(as.matrix(d), m = a$m %||% 5L, maxit = a$maxit %||% 10L, seed = a$seed %||% 0)
    imps <- lapply(r$imputations, function(x) { x <- as.data.frame(x); names(x) <- names(d); x })
    out <- data.frame(variable = names(d), n_missing = colSums(r$missing_mask),
                      pooled_mean = colMeans(do.call(rbind, lapply(imps, colMeans))),
                      row.names = NULL)
    attr(out, "imputations") <- imps
    attr(out, "method") <- "multivariate-normal MI by chained normal regressions (morie_miord2)"
    out
  },
  reference = function(a) Amelia::amelia(a$x %||% a$data, m = a$m %||% 5L, p2s = 0),
  compare = function(nat, ref) {
    rm <- colMeans(do.call(rbind, lapply(ref$imputations, colMeans)))
    list(native = nat$pooled_mean, reference = unname(rm))
  },
  tol = 0.03,
  args = .tsm_test$sim(4, {
    n <- 2000
    x1 <- rnorm(n, 10, 2); x2 <- 0.5 * x1 + rnorm(n, 0, 1.5); y <- 2 + 0.7 * x1 - 0.4 * x2 + rnorm(n)
    d <- data.frame(y = y, x1 = x1, x2 = x2)
    # one variable missing per incomplete row (Amelia leaves all-missing rows unimputed)
    i <- sample(n, 600); d$x1[i[1:200]] <- NA; d$x2[i[201:450]] <- NA; d$y[i[451:600]] <- NA
    list(x = d, m = 50L)
  }),
  note = paste("no EMB (EM + bootstrap) in morie; nearest native is proper MI under the same",
               "multivariate-normal model: morie_miord2 (chained Bayesian normal regressions, compatible",
               "with a joint MVN); pooled means agree within Monte Carlo error only - a different algorithm,",
               "not a renamed amelia (em_imputation, the EM MLE, crashes on rows with every value missing)")
)

# ------------------------------------------------------------------------------------------------
# Survey: r_svydesign / r_svymean / r_svytotal / r_svyglm / r_calibrate / r_rake
# ------------------------------------------------------------------------------------------------
.tsm_test$svy_data <- .tsm_test$sim(5, {
  n <- 400
  s <- rep(c("north", "south", "east", "west"), each = n / 4)
  psu <- paste0(s, "-", rep(1:20, each = 5))
  w <- runif(n, 5, 40)
  x <- rnorm(n, 10, 3)
  g <- factor(sample(c("m", "f"), n, TRUE))
  r <- factor(sample(c("a", "b", "c"), n, TRUE, prob = c(.5, .3, .2)))
  y <- 5 + 0.8 * x + (s == "north") * 2 + rnorm(n, 0, 2)
  yb <- rbinom(n, 1, plogis(-2 + 0.2 * x))
  data.frame(y = y, yb = yb, x = x, g = g, r = r, s = s, psu = psu, w = w)
})

ADAPTERS[["r_svydesign"]] <- list(
  native = function(a, M) {
    d <- .tsm$design(a, M)
    out <- data.frame(n = nrow(d$data), strata = length(unique(d$strata)),
                      psus = length(unique(d$cluster)), sum_weights = sum(d$weights))
    attr(out, "design") <- d
    out
  },
  reference = function(a) .tsm_test$ref_design(a),
  compare = function(nat, ref) {
    d <- attr(nat, "design")
    list(native = c(nat$n, nat$strata, nat$psus, d$weights, d$n_psu),
         reference = c(nrow(ref$variables), length(unique(ref$strata[[1]])),
                       length(unique(paste(ref$strata[[1]], ref$cluster[[1]]))),
                       1 / ref$prob, ref$fpc$sampsize[, 1]))
  },
  tol = 1e-12,
  args = list(ids = ~psu, weights = ~w, strata = ~s, data = .tsm_test$svy_data),
  note = "morie_survey_design(data, weights_col, strata_col, cluster_col); same weights, strata, PSU counts"
)

ADAPTERS[["r_svymean"]] <- list(
  native = function(a, M) {
    d <- .tsm$design(a, M)
    vars <- .tsm$fvars(a$x %||% a$formula)
    rows <- list()
    for (v in vars) {
      col <- d$data[[v]]
      if (is.numeric(col)) {
        r <- M$morie_survey_mean(d, v)
        rows[[length(rows) + 1L]] <- data.frame(term = v, mean = r$mean, SE = r$se)
      } else {
        # a factor: one proportion per level, as svymean does
        for (lv in levels(factor(col))) {
          nm <- paste0(v, lv)
          d$data[[nm]] <- as.numeric(col == lv)
          r <- M$morie_survey_mean(d, nm)
          rows[[length(rows) + 1L]] <- data.frame(term = nm, mean = r$mean, SE = r$se)
        }
      }
    }
    do.call(rbind, rows)
  },
  reference = function(a) survey::svymean(a$x %||% a$formula, .tsm_test$ref_design(a)),
  compare = function(nat, ref) list(native = c(nat$mean, nat$SE),
                                    reference = unname(c(coef(ref), survey::SE(ref)))),
  tol = 1e-10,
  args = list(x = ~y + x + g, data = .tsm_test$svy_data, weights = ~w, ids = ~psu, strata = ~s),
  note = "morie_survey_mean (Hajek mean + Taylor-linearisation SE, strata/PSU/fpc); factors done per level"
)

ADAPTERS[["r_svytotal"]] <- list(
  native = function(a, M) {
    d <- .tsm$design(a, M)
    vars <- .tsm$fvars(a$x %||% a$formula)
    rows <- list()
    for (v in vars) {
      col <- d$data[[v]]
      cols <- if (is.numeric(col)) setNames(list(as.numeric(col)), v) else
        setNames(lapply(levels(factor(col)), function(lv) as.numeric(col == lv)),
                 paste0(v, levels(factor(col))))
      for (nm in names(cols)) {
        z <- d$weights * cols[[nm]]
        # .morie_svy_recvar is the design-variance engine behind morie_survey_mean / _glm
        se <- sqrt(M$.morie_svy_recvar(cbind(z), d)[1, 1])
        rows[[length(rows) + 1L]] <- data.frame(term = nm, total = sum(z), SE = se)
      }
    }
    do.call(rbind, rows)
  },
  reference = function(a) survey::svytotal(a$x %||% a$formula, .tsm_test$ref_design(a)),
  compare = function(nat, ref) list(native = c(nat$total, nat$SE),
                                    reference = unname(c(coef(ref), survey::SE(ref)))),
  tol = 1e-10,
  args = list(x = ~y + g, data = .tsm_test$svy_data, weights = ~w, ids = ~psu, strata = ~s),
  note = paste("no exported design-based total; HT total sum(w*y) with SE from morie's internal",
               ".morie_svy_recvar (the Taylor variance used by morie_survey_mean/glm); morie_survey_ht_total",
               "is a Poisson-sampling variance and does not match svytotal")
)

ADAPTERS[["r_svyglm"]] <- list(
  native = function(a, M) {
    d <- .tsm$design(a, M)
    fam <- a$family %||% "gaussian"
    if (is.character(fam)) fam <- sub("^quasi", "", fam)
    r <- M$morie_survey_glm(d, a$formula, family = fam)
    r$coefficients
  },
  reference = function(a) survey::svyglm(a$formula, .tsm_test$ref_design(a), family = a$family %||% "gaussian"),
  compare = function(nat, ref) {
    cf <- summary(ref)$coefficients
    list(native = c(nat[, 1], nat[, 2]), reference = unname(c(cf[, 1], cf[, 2])))
  },
  tol = 1e-6,
  args = list(formula = yb ~ x + g, data = .tsm_test$svy_data, family = "quasibinomial",
              weights = ~w, ids = ~psu, strata = ~s),
  note = paste("morie_survey_glm (sandwich SE on the design); quasi* family names map to the native's",
               "binomial/poisson (same estimates); ~5e-8 rel gap is IRLS stopping (glm epsilon 1e-8), tol 1e-6")
)

ADAPTERS[["r_calibrate"]] <- list(
  native = function(a, M) {
    d <- .tsm$design(a, M)
    X <- stats::model.matrix(a$formula, d$data)
    pop <- a$population
    if (!is.null(names(pop))) pop <- pop[colnames(X)]
    calfun <- a$calfun %||% "linear"
    w <- if (identical(calfun, "linear")) {
      M$morie_weights_greg(d$weights, X, as.numeric(pop))$weights
    } else if (identical(calfun, "raking")) {
      tot <- setNames(as.list(as.numeric(pop)), colnames(X))
      M$morie_weights_calibrate_to_totals(d$weights, as.data.frame(X, check.names = FALSE), tot,
                                          method = "raking")$weights
    } else stop("native calibration supports calfun = 'linear' or 'raking'.", call. = FALSE)
    data.frame(term = colnames(X), target = as.numeric(pop), calibrated_total = colSums(X * w),
               row.names = NULL) -> out
    attr(out, "weights") <- w
    out
  },
  reference = function(a) {
    args <- list(.tsm_test$ref_design(a), a$formula, population = a$population)
    if (!is.null(a$calfun)) args$calfun <- a$calfun
    do.call(survey::calibrate, args)
  },
  compare = function(nat, ref) list(native = attr(nat, "weights"), reference = as.numeric(weights(ref))),
  tol = 1e-8,
  args = list(data = .tsm_test$svy_data, formula = ~x + g, population = c(`(Intercept)` = 10000, x = 101000, gm = 4900),
              weights = ~w, ids = ~psu, strata = ~s),
  note = paste("morie_weights_greg (linear/GREG, survey's default calfun; 1e-15);",
               "calfun='raking' -> morie_weights_calibrate_to_totals (~4e-9, survey's Newton stops at epsilon 1e-7)")
)

ADAPTERS[["r_rake"]] <- list(
  native = function(a, M) {
    d <- .tsm$design(a, M)
    sm <- a$sample.margins
    pm <- a$population.margins
    if (inherits(sm, "formula")) sm <- list(sm)
    if (is.data.frame(pm) || is.table(pm)) pm <- list(pm)
    margins <- list()
    for (i in seq_along(sm)) {
      v <- .tsm$fvars(sm[[i]])
      if (length(v) != 1L) stop("each sample margin must be one variable (~g).", call. = FALSE)
      p <- pm[[i]]
      margins[[v]] <- if (is.data.frame(p)) setNames(as.numeric(p[[ncol(p)]]), as.character(p[[v]]))
                      else setNames(as.numeric(p), names(p))
    }
    r <- M$morie_rake(d$data, margins, weights = d$weights, max_iter = 1000L, tol = 1e-10)
    out <- do.call(rbind, lapply(names(margins), function(v) {
      cur <- tapply(r$weights, factor(as.character(d$data[[v]]), levels = names(margins[[v]])), sum)
      data.frame(margin = v, level = names(margins[[v]]), target = unname(margins[[v]]),
                 raked_total = as.numeric(cur))
    }))
    attr(out, "weights") <- r$weights
    attr(out, "converged") <- r$converged
    out
  },
  reference = function(a) {
    # survey's default control (epsilon = 1, i.e. stop once the joint table moves by < 1 unit of
    # weight; maxit = 10) stops early; ask it to converge so the fixed point can be compared
    survey::rake(.tsm_test$ref_design(a), a$sample.margins, a$population.margins,
                 control = list(maxit = 1000, epsilon = 1e-13, verbose = FALSE))
  },
  compare = function(nat, ref) list(native = attr(nat, "weights"), reference = as.numeric(weights(ref))),
  tol = 1e-8,
  args = list(data = .tsm_test$svy_data, sample.margins = list(~g, ~r),
              population.margins = list(data.frame(g = c("f", "m"), Freq = c(5100, 4900)),
                                        data.frame(r = c("a", "b", "c"), Freq = c(4500, 3500, 2000))),
              weights = ~w, ids = ~psu, strata = ~s),
  note = paste("morie_rake (IPF to tol 1e-10); matches survey::rake run to convergence; survey's default",
               "control (epsilon = 1 weight unit, maxit = 10) stops earlier, so default-run weights differ slightly")
)

# ------------------------------------------------------------------------------------------------
# r_boot / r_boot_ci: boot::boot, boot::boot.ci  ->  morie_boot, morie_boot_ci (same RNG stream)
# ------------------------------------------------------------------------------------------------
ADAPTERS[["r_boot"]] <- list(
  native = function(a, M) {
    if (!is.null(a$seed)) set.seed(a$seed)
    b <- M$morie_boot(a$data, .tsm$stat_fun(a$statistic), R = a$R %||% 1000)
    out <- data.frame(statistic = a$statistic, original = b$t0, bias = mean(b$t[, 1]) - b$t0,
                      std.error = stats::sd(b$t[, 1]), R = b$R)
    attr(out, "boot") <- b
    out
  },
  reference = function(a) {
    if (!is.null(a$seed)) set.seed(a$seed)
    boot::boot(a$data, .tsm$stat_fun(a$statistic), R = a$R %||% 1000)
  },
  compare = function(nat, ref) list(native = c(nat$original, nat$bias, nat$std.error, attr(nat, "boot")$t),
                                    reference = c(ref$t0, mean(ref$t[, 1]) - ref$t0, stats::sd(ref$t[, 1]), ref$t)),
  tol = 1e-12,
  args = .tsm_test$sim(6, list(data = rexp(200, 0.2), statistic = "median", R = 2000L, seed = 2026L)),
  note = paste("morie_boot reproduces boot's index stream: identical replicates under a common seed",
               "(seed= is an adapter extra; without it both are random and agree only within MC error)")
)

ADAPTERS[["r_boot_ci"]] <- list(
  native = function(a, M) {
    if (!is.null(a$seed)) set.seed(a$seed)
    b <- M$morie_boot(a$data, .tsm$stat_fun(a$statistic), R = a$R %||% 1000)
    type <- a$type %||% "perc"
    if (identical(type, "all")) type <- c("norm", "basic", "perc", "bca")
    if ("stud" %in% type) stop("studentized intervals need bootstrap variances; not available.", call. = FALSE)
    ci <- M$morie_boot_ci(b, conf = a$conf %||% 0.95, type = type)
    data.frame(type = names(ci), conf = a$conf %||% 0.95, lower = vapply(ci, `[`, 0, 1),
               upper = vapply(ci, `[`, 0, 2), row.names = NULL)
  },
  reference = function(a) {
    if (!is.null(a$seed)) set.seed(a$seed)
    b <- boot::boot(a$data, .tsm$stat_fun(a$statistic), R = a$R %||% 1000)
    boot::boot.ci(b, conf = a$conf %||% 0.95, type = a$type %||% "perc")
  },
  compare = function(nat, ref) {
    key <- c(norm = "normal", basic = "basic", perc = "percent", bca = "bca")
    rv <- unlist(lapply(nat$type, function(t) {
      m <- ref[[key[[t]]]]
      m[1, ncol(m) - 1:0]
    }))
    list(native = c(rbind(nat$lower, nat$upper)), reference = unname(rv))
  },
  tol = 1e-12,
  args = .tsm_test$sim(7, list(data = rexp(200, 0.2), statistic = "mean", R = 2000L,
                          type = c("norm", "basic", "perc", "bca"), seed = 99L)),
  note = "morie_boot + morie_boot_ci (norm/basic/perc/bca, boot's formulas and BCa empinf.reg); 'stud' not supported"
)

# ------------------------------------------------------------------------------------------------
# r_metafor(yi, vi, method): metafor::rma  ->  random_effects_meta / fixed_effects_meta
# ------------------------------------------------------------------------------------------------
ADAPTERS[["r_metafor"]] <- list(
  native = function(a, M) {
    method <- a$method %||% "REML"
    se <- sqrt(a$vi)
    if (method %in% c("FE", "EE", "CE")) {
      r <- M$fixed_effects_meta(a$yi, se, confidence = a$level %||% 0.95)
      tau2 <- 0
    } else if (method %in% c("DL", "PM", "REML")) {
      r <- M$random_effects_meta(a$yi, se, confidence = a$level %||% 0.95, method = method)
      tau2 <- r$extra$tau_squared
    } else stop("native tau^2 estimators: DL, PM, REML (or FE/EE).", call. = FALSE)
    z <- r$estimate / r$se
    data.frame(method = method, estimate = r$estimate, se = r$se, zval = z,
               pval = 2 * stats::pnorm(-abs(z)), ci.lb = r$ci_lower, ci.ub = r$ci_upper,
               tau2 = tau2, Q = r$extra$Q, I2 = r$extra$I_squared %||% NA_real_)
  },
  reference = function(a) {
    # tighter than rma's default Fisher-scoring threshold (1e-5 on tau^2), so the REML fixed point
    # is compared rather than rma's stopping rule
    metafor::rma(yi = a$yi, vi = a$vi, method = a$method %||% "REML",
                 control = list(threshold = 1e-12, tol = 1e-14, maxiter = 1000))
  },
  # I2 is left out: rma reports tau2-based I2 = tau2/(tau2 + s2), morie the Q-based (Q-df)/Q
  compare = function(nat, ref) list(native = c(nat$estimate, nat$se, nat$ci.lb, nat$ci.ub, nat$tau2, nat$Q, nat$pval),
                                    reference = c(ref$b[1], ref$se, ref$ci.lb, ref$ci.ub, ref$tau2, ref$QE, ref$pval)),
  tol = 1e-7,
  args = .tsm_test$sim(8, {
    k <- 15
    vi <- runif(k, 0.02, 0.10)
    list(yi = rnorm(k, 0.3, sqrt(vi + 0.04)), vi = vi, method = "REML")
  }),
  note = paste("random_effects_meta(estimates, sqrt(vi), method): same Wald SE 1/sqrt(sum w*), no Knapp-Hartung;",
               "the earlier SE gap was rma's own REML stopping rule (threshold 1e-5 on tau^2: ~2e-5 rel in tau2,",
               "~1e-6 rel in SE; PM likewise via rma's root-finder tol); with tight control they agree.",
               "I2 differs for REML/PM: rma uses tau2-based I2, morie the Q-based (Higgins) one")
)

# ------------------------------------------------------------------------------------------------
# r_meta_bin(event.e, n.e, event.c, n.c): meta::metabin  ->  malab/Mamh (MH common) + malrr/malor
#                                                           + random_effects_meta (random effects)
# ------------------------------------------------------------------------------------------------
ADAPTERS[["r_meta_bin"]] <- list(
  native = function(a, M) {
    ee <- a$event.e; ne <- a$n.e; ec <- a$event.c; nc <- a$n.c
    sm <- a$sm %||% "RR"
    tau_m <- a$method.tau %||% "REML"
    if (sm == "RR") {
      common <- M$malab(ee, ne, ec, nc)$log_rr
      common_se <- NA_real_  # morie has no Greenland-Robins variance for the MH risk ratio
      st <- M$malrr(ee, ne - ee, ec, nc - ec)
    } else if (sm == "OR") {
      mh <- M$Mamh(ee, ne - ee, ec, nc - ec)
      common <- log(mh$estimate); common_se <- mh$se
      st <- M$malor(ee, ne - ee, ec, nc - ec)
    } else stop("native summary measures: sm = 'RR' or 'OR'.", call. = FALSE)
    re <- M$random_effects_meta(st$estimate, sqrt(st$variance), method = tau_m)
    data.frame(model = c("common (Mantel-Haenszel)", sprintf("random (%s)", tau_m)), sm = sm,
               log_estimate = c(common, re$estimate), se = c(common_se, re$se),
               estimate = exp(c(common, re$estimate)),
               tau2 = c(NA, re$extra$tau_squared), row.names = NULL)
  },
  reference = function(a) {
    args <- list(a$event.e, a$n.e, a$event.c, a$n.c)
    for (k in c("sm", "method.tau")) if (!is.null(a[[k]])) args[[k]] <- a[[k]]
    do.call(meta::metabin, args)
  },
  compare = function(nat, ref) list(native = c(nat$log_estimate[1], nat$log_estimate[2], nat$se[2], nat$tau2[2]),
                                    reference = c(ref$TE.common, ref$TE.random, ref$seTE.random, ref$tau2)),
  tol = 1e-4,
  args = .tsm_test$sim(9, {
    k <- 10
    ne <- sample(60:200, k); nc <- sample(60:200, k)
    list(event.e = rbinom(k, ne, 0.30), n.e = ne, event.c = rbinom(k, nc, 0.20), n.c = nc)
  }),
  note = paste("MH risk ratio (malab) for the common effect, per-study log RR (malrr) pooled by",
               "random_effects_meta(REML) for the random effect; morie lacks the MH-RR (Greenland-Robins)",
               "SE, so the common-effect SE/CI is missing (sm='OR' has it via Mamh); tol 1e-4 because",
               "meta's REML goes through rma's default 1e-5 stopping threshold")
)

# ------------------------------------------------------------------------------------------------
# Plots: r_ggplot / r_ggsurvplot / r_forestplot. morie's own plotting is base graphics
# (morie_eda_plot); the native adapters draw with it / base graphics into a PNG.
# compare() returns what was drawn (points, strata, curve values) against the data.
# ------------------------------------------------------------------------------------------------
.tsm$open_png <- function(a) {
  file <- a$file %||% tempfile("morie-plot-", fileext = ".png")
  grDevices::png(file, width = 1600, height = 1000, res = 150)
  file
}

ADAPTERS[["r_ggplot"]] <- list(
  native = function(a, M) {
    d <- a$data; g <- a$geom %||% "point"
    geoms <- c("point", "line", "col", "bar", "histogram", "boxplot", "density", "smooth")
    if (!g %in% geoms) stop("geom must be one of: ", paste(geoms, collapse = ", "), call. = FALSE)
    file <- .tsm$open_png(a)
    on.exit(grDevices::dev.off())
    x <- d[[a$x]]; y <- if (is.null(a$y)) NULL else d[[a$y]]
    drawn <- switch(g,
      point = nrow(M$morie_eda_plot(d, a$x, a$y)),
      line = { o <- order(x); graphics::plot(x[o], y[o], type = "l", xlab = a$x, ylab = a$y); length(x) },
      col = { graphics::barplot(y, names.arg = as.character(x), xlab = a$x, ylab = a$y); length(y) },
      bar = { tb <- table(x); graphics::barplot(tb, xlab = a$x, ylab = "count"); length(tb) },
      histogram = { h <- graphics::hist(x, breaks = 30, main = "", xlab = a$x); sum(h$counts) },
      boxplot = {
        if (is.null(y)) { graphics::boxplot(x, ylab = a$x); 1L }
        else { b <- graphics::boxplot(y ~ x, xlab = a$x, ylab = a$y); ncol(b$stats) }
      },
      density = { de <- stats::density(x); graphics::plot(de, main = "", xlab = a$x); length(de$x) },
      smooth = {
        graphics::plot(x, y, xlab = a$x, ylab = a$y, pch = 19, col = "grey60")
        fit <- stats::loess(y ~ x); xs <- seq(min(x), max(x), length.out = 80)
        graphics::lines(xs, stats::predict(fit, data.frame(x = xs)), lwd = 2); 80L
      })
    cat("plot saved to", file, "\n")
    invisible(list(file = file, geom = g, drawn = drawn))
  },
  reference = function(a) {
    map <- if (is.null(a$y)) ggplot2::aes(x = .data[[a$x]]) else ggplot2::aes(x = .data[[a$x]], y = .data[[a$y]])
    p <- ggplot2::ggplot(a$data, map) + getExportedValue("ggplot2", paste0("geom_", a$geom %||% "point"))()
    ld <- suppressMessages(ggplot2::layer_data(p))
    g <- a$geom %||% "point"
    list(geom = g, drawn = switch(g, histogram = sum(ld$count), nrow(ld)))
  },
  compare = function(nat, ref) list(native = nat$drawn, reference = ref$drawn),
  tol = 0,
  args = .tsm_test$sim(10, list(data = data.frame(hp = runif(120, 50, 300), grp = factor(sample(letters[1:4], 120, TRUE)),
                                              mpg = rnorm(120, 25, 5)),
                           x = "hp", y = "mpg", geom = "point")),
  note = paste("morie_eda_plot (base graphics) for geom='point'; other geoms drawn with base graphics",
               "(morie has no line/hist/box/density plotters); compare = number of points/bars/boxes/",
               "observations drawn vs ggplot's layer data")
)

ADAPTERS[["r_ggsurvplot"]] <- list(
  native = function(a, M) {
    p <- .tsm$km_parts(a$formula, a$data)
    lv <- levels(p$group)
    fits <- lapply(lv, function(l) {
      i <- p$group == l
      M$morie_kaplan_meier(p$time[i], p$event[i])
    })
    names(fits) <- lv
    file <- .tsm$open_png(a)
    on.exit(grDevices::dev.off())
    cols <- c("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")
    graphics::plot(NA, xlim = c(0, max(p$time)), ylim = c(0, 1), xlab = "Time", ylab = "Survival probability")
    for (j in seq_along(fits)) {
      f <- fits[[j]]
      graphics::lines(c(0, f$time, max(p$time[p$group == lv[j]])), c(1, f$surv, utils::tail(c(1, f$surv), 1)),
                      type = "s", col = cols[(j - 1) %% length(cols) + 1], lwd = 2)
    }
    if (length(lv) > 1) graphics::legend("topright", legend = lv, col = cols[seq_along(lv)], lwd = 2, bty = "n")
    cat("plot saved to", file, "\n")
    invisible(list(file = file, strata = lv, km = fits))
  },
  reference = function(a) {
    fit <- do.call(survival::survfit, list(formula = a$formula, data = a$data))
    file <- tempfile("ref-plot-", fileext = ".png")
    grDevices::png(file, width = 1600, height = 1000, res = 150)
    on.exit(grDevices::dev.off())
    print(survminer::ggsurvplot(fit, data = a$data))
    fit
  },
  compare = function(nat, ref) {
    # the curve values drawn: S(t) at each event time, per stratum
    ss <- summary(ref)
    list(native = c(length(nat$strata), unlist(lapply(nat$km, `[[`, "surv")), unlist(lapply(nat$km, `[[`, "n_risk"))),
         reference = c(max(1, length(ref$strata)), ss$surv, ss$n.risk))
  },
  tol = 1e-12,
  args = .tsm_test$sim(11, {
    n <- 200
    grp <- factor(sample(c("ctrl", "trt"), n, TRUE))
    t <- rexp(n, ifelse(grp == "trt", 0.05, 0.09)); cens <- runif(n, 0, 40)
    list(formula = Surv(time, status) ~ grp,
         data = data.frame(time = round(pmin(t, cens), 1), status = as.numeric(t <= cens), grp = grp))
  }),
  note = paste("morie_kaplan_meier per stratum, drawn as base-graphics step curves (morie has no KM",
               "plotter); the drawn S(t) and n at risk equal survfit's exactly")
)

ADAPTERS[["r_forestplot"]] <- list(
  native = function(a, M) {
    x <- a$data
    lab <- as.character(x[[a$label %||% "label"]]); mn <- x[[a$mean %||% "mean"]]
    lo <- x[[a$lower %||% "lower"]]; hi <- x[[a$upper %||% "upper"]]
    k <- length(mn)
    file <- .tsm$open_png(a)
    on.exit(grDevices::dev.off())
    op <- graphics::par(mar = c(4, max(8, max(nchar(lab)) * 0.6), 1, 1)); on.exit(graphics::par(op), add = TRUE)
    yy <- rev(seq_len(k))
    ok <- is.finite(mn)
    graphics::plot(mn[ok], yy[ok], xlim = range(c(lo, hi), na.rm = TRUE), ylim = c(0.5, k + 0.5), pch = 15,
                   yaxt = "n", ylab = "", xlab = "estimate")
    graphics::segments(lo[ok], yy[ok], hi[ok], yy[ok])
    graphics::axis(2, at = yy, labels = lab, las = 1, tick = FALSE)
    graphics::abline(v = 0, lty = 2, col = "grey50")
    cat("plot saved to", file, "\n")
    invisible(list(file = file, rows = k, drawn = sum(ok), mean = mn, lower = lo, upper = hi))
  },
  reference = function(a) {
    x <- a$data
    file <- tempfile("ref-plot-", fileext = ".png")
    grDevices::png(file, width = 1600, height = 1000, res = 150)
    on.exit(grDevices::dev.off())
    fp <- forestplot::forestplot(labeltext = as.character(x[[a$label %||% "label"]]), mean = x[[a$mean %||% "mean"]],
                                 lower = x[[a$lower %||% "lower"]], upper = x[[a$upper %||% "upper"]])
    print(fp)
    list(rows = nrow(x), drawn = sum(is.finite(x[[a$mean %||% "mean"]])), mean = x[[a$mean %||% "mean"]],
         lower = x[[a$lower %||% "lower"]], upper = x[[a$upper %||% "upper"]])
  },
  compare = function(nat, ref) list(native = c(nat$rows, nat$drawn, nat$mean, nat$lower, nat$upper),
                                    reference = c(ref$rows, ref$drawn, ref$mean, ref$lower, ref$upper)),
  tol = 0,
  args = .tsm_test$sim(12, {
    m <- rnorm(8, 0.2, 0.3); s <- runif(8, 0.05, 0.2)
    list(data = data.frame(label = paste("Study", 1:8), mean = m, lower = m - 1.96 * s, upper = m + 1.96 * s))
  }),
  note = paste("no native forest-plot drawer in morie (Magal gives Galbraith/radial coordinates only);",
               "drawn with base graphics; compare = rows/intervals drawn vs the data")
)
