# Adapters for the `regression` group of morie's R-bridge commands.
# Each `native` calls morie's own R functions (via M$...); each `reference` is the outside /
# stats function the bridge calls today, used only to validate the numbers.

if (!exists("ADAPTERS")) ADAPTERS <- list()

# ---------------------------------------------------------------------------------------------
# shared helpers (argument conversion only; all fitting is done by morie functions)
# ---------------------------------------------------------------------------------------------

.rg_coef_table <- function(est, se = NULL, stat_name = NULL, df = NULL) {
  out <- data.frame(Estimate = as.numeric(est), row.names = names(est))
  if (!is.null(se)) {
    out[["Std.Error"]] <- as.numeric(se)
    z <- as.numeric(est) / as.numeric(se)
    if (identical(stat_name, "t")) {
      out[["t value"]] <- z
      out[["Pr(>|t|)"]] <- 2 * stats::pt(abs(z), df, lower.tail = FALSE)
    } else if (identical(stat_name, "z")) {
      out[["z value"]] <- z
      out[["Pr(>|z|)"]] <- 2 * stats::pnorm(abs(z), lower.tail = FALSE)
    }
  }
  out
}

# model frame -> numeric response + design matrix (factors expanded with R's contrasts)
.rg_design <- function(formula, data) {
  mf <- stats::model.frame(formula, data)
  y <- stats::model.response(mf)
  X <- stats::model.matrix(attr(mf, "terms"), mf)
  list(y = y, X = X, mf = mf)
}

# a 0/1 response as glm(binomial) codes it: factor -> first level is failure
.rg_binary <- function(y) {
  if (is.factor(y)) return(as.numeric(y != levels(y)[1]))
  if (is.logical(y)) return(as.numeric(y))
  as.numeric(y)
}

.rg_family_name <- function(family) {
  if (is.null(family)) return("gaussian")
  if (is.function(family)) family <- family()
  if (inherits(family, "family")) {
    fam <- tolower(family$family); link <- family$link
  } else {
    fam <- tolower(as.character(family)[1])
    link <- switch(fam, binomial = "logit", poisson = "log", gaussian = "identity",
                   gamma = "inverse", quasibinomial = "logit", quasipoisson = "log", NA)
  }
  canon <- c(binomial = "logit", poisson = "log", gaussian = "identity", gamma = "log")
  if (!fam %in% names(canon))
    stop("morie_glm supports the binomial, poisson, gaussian and Gamma(log) families, not '", fam, "'",
         call. = FALSE)
  if (!identical(link, canon[[fam]]))
    stop(sprintf("morie_glm fits family '%s' with the %s link only (got link '%s')", fam,
                 canon[[fam]], link), call. = FALSE)
  fam
}

.rg_glm_native <- function(formula, data, family, M) {
  fam <- .rg_family_name(family)
  d <- .rg_design(formula, data)
  y <- if (fam == "binomial") .rg_binary(d$y) else as.numeric(d$y)
  # morie_glm's defaults (tol 1e-8 on the relative deviance change, 25 iterations) are glm's
  r <- M$morie_glm(y, d$X, family = fam, add_intercept = FALSE)
  names(r$coef) <- colnames(d$X)
  list(call = sprintf("morie_glm(family = \"%s\")", fam),
       coefficients = .rg_coef_table(setNames(r$coef, colnames(d$X)), r$se, r$statistic_name,
                                     r$df_residual),
       dispersion = r$dispersion, deviance = r$deviance, null_deviance = r$null_deviance,
       df_residual = r$df_residual, aic = r$aic, converged = r$converged, n = r$n)
}

.rg_glm_compare <- function(nat, ref) {
  s <- summary(ref)$coefficients
  list(native = c(nat$coefficients$Estimate, nat$coefficients$Std.Error, nat$deviance),
       reference = c(s[, 1], s[, 2], ref$deviance))
}

# --- mixed-model formulas: split y ~ x + (1 + x | g) into fixed part and bar terms -----------
.rg_split_bars <- function(expr) {
  bars <- list()
  walk <- function(e) {
    if (is.call(e) && identical(e[[1]], as.name("+")) && length(e) == 3) {
      l <- walk(e[[2]]); r <- walk(e[[3]])
      if (is.null(l)) return(r)
      if (is.null(r)) return(l)
      return(call("+", l, r))
    }
    if (is.call(e) && identical(e[[1]], as.name("("))) e <- e[[2]]
    if (is.call(e) && identical(e[[1]], as.name("|"))) {
      bars[[length(bars) + 1L]] <<- e
      return(NULL)
    }
    e
  }
  fixed <- walk(expr)
  list(fixed = if (is.null(fixed)) 1 else fixed, bars = bars)
}

# random-effects design: one block per bar term, columns ordered group-major
.rg_random_design <- function(bars, data) {
  lapply(bars, function(b) {
    lhs <- b[[2]]
    g <- factor(eval(b[[3]], data, environment()))
    Zt <- stats::model.matrix(stats::as.formula(call("~", lhs)), data)
    m <- nlevels(g); r <- ncol(Zt)
    Z <- matrix(0, nrow(data), m * r)
    gi <- as.integer(g)
    for (k in seq_len(r)) Z[cbind(seq_len(nrow(data)), (gi - 1L) * r + k)] <- Zt[, k]
    list(Z = Z, m = m, r = r, names = colnames(Zt), group = deparse(b[[3]]), g = g)
  })
}

.rg_chol_from <- function(par, r) {
  L <- matrix(0, r, r)
  L[lower.tri(L, diag = TRUE)] <- par
  diag(L) <- exp(diag(L))
  L
}

# REML (or ML) LMM fit composed from morie natives: Remlik / morie_lmm_loglik give the
# (restricted) log-likelihood and GLS beta for given covariances, NelderMead maximises it.
.rg_lmm_native <- function(fixed, bars, data, M, reml = TRUE, restarts = 1L) {
  mf_vars <- unique(c(all.vars(fixed), unlist(lapply(bars, all.vars))))
  data <- data[stats::complete.cases(data[mf_vars]), , drop = FALSE]
  d <- .rg_design(fixed, data)
  y <- as.numeric(d$y); X <- d$X
  blocks <- .rg_random_design(bars, data)
  Z <- do.call(cbind, lapply(blocks, `[[`, "Z"))
  n <- length(y)
  npar <- vapply(blocks, function(b) b$r * (b$r + 1L) / 2L, 1)
  build <- function(th) {
    s2 <- exp(2 * th[1]); pos <- 1L
    Ds <- lapply(seq_along(blocks), function(i) {
      b <- blocks[[i]]
      L <- .rg_chol_from(th[pos + seq_len(npar[i])], b$r); pos <<- pos + npar[i]
      list(S = L %*% t(L), D = kronecker(diag(b$m), L %*% t(L)))
    })
    q <- ncol(Z); D <- matrix(0, q, q); off <- 0L
    for (x in Ds) { k <- nrow(x$D); D[off + seq_len(k), off + seq_len(k)] <- x$D; off <- off + k }
    list(s2 = s2, D = D, S = lapply(Ds, `[[`, "S"))
  }
  obj <- function(th) {
    p <- build(th)
    r <- tryCatch(if (reml) M$Remlik(X, Z, y, p$D, R = diag(p$s2, n))$loglik
                  else M$morie_lmm_loglik(X, Z, y, p$D, R = diag(p$s2, n))$loglik,
                  error = function(e) -Inf)
    if (!is.finite(r)) 1e100 else -r
  }
  ols <- M$morie_ols(y, X, add_intercept = FALSE)
  s0 <- sqrt(ols$rss / ols$df_resid)
  th0 <- c(log(s0 / sqrt(2)), unlist(lapply(blocks, function(b) {
    L <- diag(log(s0 / sqrt(2)), b$r); L[lower.tri(L, diag = TRUE)]
  })))
  # Nelder-Mead, restarted from its own optimum until the objective stops moving (a restart
  # rebuilds a fresh simplex, the standard guard against a collapsed one)
  opt <- M$NelderMead(obj, th0, xtol = 1e-8, ftol = 1e-12, max_iter = 20000)
  for (k in seq_len(restarts)) {
    prev <- opt$fun
    opt <- M$NelderMead(obj, opt$x, xtol = 1e-9, ftol = 1e-13, max_iter = 20000)
    if (prev - opt$fun < 1e-10) break
  }
  p <- build(opt$x)
  V <- M$morie_lmm_v(Z, p$D, diag(p$s2, n))
  Vi <- M$morie_solve(V)
  vb <- M$morie_solve(crossprod(X, Vi %*% X))
  beta <- as.numeric(vb %*% crossprod(X, Vi %*% y))
  names(beta) <- colnames(X)
  vc <- do.call(rbind, lapply(seq_along(blocks), function(i) {
    b <- blocks[[i]]; S <- p$S[[i]]
    rows <- data.frame(group = b$group, var1 = b$names, var2 = NA_character_,
                       vcov = diag(S), sdcor = sqrt(diag(S)), stringsAsFactors = FALSE)
    if (b$r > 1) {
      idx <- which(lower.tri(S), arr.ind = TRUE)
      R <- stats::cov2cor(S)
      rows <- rbind(rows, data.frame(group = b$group, var1 = b$names[idx[, 2]],
                                     var2 = b$names[idx[, 1]], vcov = S[idx], sdcor = R[idx],
                                     stringsAsFactors = FALSE))
    }
    rows
  }))
  vc <- rbind(vc, data.frame(group = "Residual", var1 = NA_character_, var2 = NA_character_,
                             vcov = p$s2, sdcor = sqrt(p$s2), stringsAsFactors = FALSE))
  list(method = if (reml) "REML (morie Remlik + NelderMead)" else "ML (morie_lmm_loglik + NelderMead)",
       fixed = .rg_coef_table(beta, sqrt(diag(vb)), "t", n - ncol(X)),
       varcomp = vc, loglik = -opt$fun, converged = opt$converged, n = n,
       n_groups = vapply(blocks, `[[`, 1, "m"))
}

.rg_merMod_vc <- function(f) {
  v <- as.data.frame(lme4::VarCorr(f))
  v$vcov
}

# ---------------------------------------------------------------------------------------------
# example data (realistic, seeded, large enough that the statistics are stable)
# ---------------------------------------------------------------------------------------------
.rg_data <- local({
  set.seed(20261010)
  n <- 300
  x1 <- rnorm(n); x2 <- runif(n, -1, 1)
  f <- factor(sample(c("a", "b", "c"), n, replace = TRUE))
  g <- factor(rep(sprintf("g%02d", 1:30), each = 10))
  b0 <- rnorm(30, sd = 0.8)[as.integer(g)]
  b1 <- rnorm(30, sd = 0.4)[as.integer(g)]
  y <- 1 + 0.7 * x1 - 0.5 * x2 + c(a = 0, b = 0.4, c = -0.3)[as.character(f)] + rnorm(n)
  yb <- factor(ifelse(runif(n) < plogis(-0.3 + 0.9 * x1 - 0.6 * x2), "yes", "no"))
  yc <- rpois(n, exp(0.4 + 0.5 * x1 + 0.3 * x2))
  ymix <- 2 + 0.6 * x1 + b0 + b1 * x1 + rnorm(n, sd = 0.9)
  ybmix <- rbinom(n, 1, plogis(-0.2 + 0.8 * x1 + b0))
  ycmix <- rpois(n, exp(0.3 + 0.4 * x1 + 0.6 * b0))
  xn <- runif(n, 0, 10)
  ynls <- 8 * (1 - exp(-0.35 * xn)) + rnorm(n, sd = 0.4)
  xs <- runif(n, 0, 1)
  ys <- sin(2 * pi * xs) + 0.5 * xs + rnorm(n, sd = 0.3)
  yr <- 1 + 2 * x1 - x2 + rt(n, df = 3)
  yr[1:10] <- yr[1:10] + 15
  data.frame(y, x1, x2, f, g, yb, yc, ymix, ybmix, ycmix, xn, ynls, xs, ys, yr)
})

# ---------------------------------------------------------------------------------------------
# r_lm(formula, data)
# ---------------------------------------------------------------------------------------------
ADAPTERS[["r_lm"]] <- list(
  native = function(a, M) {
    d <- .rg_design(a$formula, a$data)
    icpt <- attr(attr(d$mf, "terms"), "intercept") == 1L
    X <- if (icpt) d$X[, -1L, drop = FALSE] else d$X
    r <- M$morie_ols(as.numeric(d$y), X, add_intercept = icpt)
    beta <- setNames(r$coef, colnames(d$X))
    list(call = "morie_ols(y, model.matrix(formula, data))",
         coefficients = .rg_coef_table(beta, r$se, "t", r$df_resid),
         sigma = r$sigma, r.squared = r$r_squared, adj.r.squared = r$adj_r_squared,
         fstatistic = c(value = r$f_statistic, numdf = r$df_model, dendf = r$df_resid), n = r$n)
  },
  reference = function(a) stats::lm(a$formula, data = a$data),
  compare = function(nat, ref) {
    s <- summary(ref)
    list(native = c(nat$coefficients$Estimate, nat$coefficients$Std.Error, nat$sigma, nat$r.squared,
                    nat$fstatistic[["value"]]),
         reference = c(s$coefficients[, 1], s$coefficients[, 2], s$sigma, s$r.squared, s$fstatistic[["value"]]))
  },
  tol = 1e-9,
  args = list(formula = y ~ x1 + x2 + f, data = .rg_data),
  note = "morie_ols on model.matrix(formula) (factor contrasts as lm); coef, SE, sigma, R2 identical up to solve() vs QR rounding"
)

# ---------------------------------------------------------------------------------------------
# r_glm(formula, data, family); r_logistic; r_poisson
# ---------------------------------------------------------------------------------------------
ADAPTERS[["r_glm"]] <- list(
  native = function(a, M) .rg_glm_native(a$formula, a$data, a$family, M),
  reference = function(a) stats::glm(a$formula, family = a$family, data = a$data),
  compare = .rg_glm_compare,
  tol = 1e-7,
  args = list(formula = y ~ x1 + x2 + f, data = .rg_data, family = "gaussian"),
  note = "morie_glm (IRLS) on model.matrix; families binomial/poisson/gaussian/Gamma(log) only -- R's default Gamma (inverse link) is refused with a clear error"
)

ADAPTERS[["r_logistic"]] <- list(
  native = function(a, M) .rg_glm_native(a$formula, a$data, "binomial", M),
  reference = function(a) stats::glm(a$formula, family = "binomial", data = a$data),
  compare = .rg_glm_compare,
  tol = 1e-7,
  args = list(formula = yb ~ x1 + x2 + f, data = .rg_data),
  note = "morie_glm(family='binomial'); factor response coded as glm does (first level = 0); both IRLS, differ only at the convergence tolerance"
)

ADAPTERS[["r_poisson"]] <- list(
  native = function(a, M) .rg_glm_native(a$formula, a$data, "poisson", M),
  reference = function(a) stats::glm(a$formula, family = "poisson", data = a$data),
  compare = .rg_glm_compare,
  tol = 1e-7,
  args = list(formula = yc ~ x1 + x2 + f, data = .rg_data),
  note = "morie_glm(family='poisson'); IRLS, agrees with glm to its convergence tolerance"
)

# ---------------------------------------------------------------------------------------------
# r_nls(formula, data, start)
# ---------------------------------------------------------------------------------------------
ADAPTERS[["r_nls"]] <- list(
  ship = FALSE,  # evaluates the model formula at trial parameters; the bridge keeps stats::nls (part of R)
  native = function(a, M) {
    start <- unlist(a$start)
    pn <- names(start)
    rhs <- a$formula[[3]]
    yv <- as.numeric(eval(a$formula[[2]], a$data, environment(a$formula)))
    # nlsgn differentiates model(x, theta) by central differences and needs length(x) == n,
    # so x is the row index and the formula is evaluated on those rows of the data
    dat <- as.list(a$data)
    model <- function(x, theta) {
      env <- lapply(dat, `[`, x)
      env[pn] <- as.list(theta)
      as.numeric(eval(rhs, env, environment(a$formula)))
    }
    r <- M$nlsgn(model, seq_along(yv), yv, start)
    beta <- setNames(r$coefficients, pn)
    df <- length(yv) - length(pn)
    list(call = "nlsgn(model, data, y, start)",
         coefficients = .rg_coef_table(beta, r$se, "t", df),
         sigma = r$sigma, rss = r$rss, iterations = r$iterations, converged = r$converged)
  },
  reference = function(a) stats::nls(a$formula, data = a$data, start = a$start),
  compare = function(nat, ref) {
    s <- summary(ref)
    list(native = c(nat$coefficients$Estimate, nat$coefficients$Std.Error, nat$sigma),
         reference = c(s$coefficients[, 1], s$coefficients[, 2], s$sigma))
  },
  tol = 1e-6,
  args = list(formula = ynls ~ A * (1 - exp(-k * xn)), data = .rg_data, start = list(A = 5, k = 0.2)),
  note = "nlsgn (Gauss-Newton, nls's convergence criterion); Jacobian by central differences vs nls's forward differences, so SEs agree to ~1e-7"
)

# ---------------------------------------------------------------------------------------------
# r_lme(fixed, data, random) -- nlme::lme, REML
# ---------------------------------------------------------------------------------------------
ADAPTERS[["r_lme"]] <- list(
  ship = FALSE,  # dense n x n fit (about 14 s at n = 300, cubic in n); the bridge keeps nlme::lme until morie has a sparse mixed-model fitter
  native = function(a, M) {
    rnd <- a$random
    rexpr <- if (inherits(rnd, "formula")) rnd[[length(rnd)]] else rnd
    if (!(is.call(rexpr) && identical(rexpr[[1]], as.name("|"))))
      stop("random must be a formula of the form ~ terms | group", call. = FALSE)
    fit <- .rg_lmm_native(a$fixed, list(rexpr), a$data, M, reml = TRUE)
    fit
  },
  reference = function(a) nlme::lme(fixed = a$fixed, data = a$data, random = a$random),
  compare = function(nat, ref) {
    vc <- nlme::getVarCov(ref)
    list(native = c(nat$fixed$Estimate, nat$fixed$Std.Error,
                    nat$varcomp$vcov[is.na(nat$varcomp$var2) & nat$varcomp$group != "Residual"],
                    nat$varcomp$vcov[nat$varcomp$group == "Residual"]),
         reference = c(nlme::fixef(ref), sqrt(diag(stats::vcov(ref))), diag(vc), ref$sigma^2))
  },
  tol = 1e-4,
  args = list(fixed = ymix ~ x1 + x2, data = .rg_data, random = ~ 1 + x1 | g),
  note = "REML maximised by morie NelderMead over morie Remlik (GLS beta, vcov from morie_lmm_v/morie_solve); random intercept+slope; lme stops at its own optimiser tolerance (~1e-6..1e-5)"
)

# ---------------------------------------------------------------------------------------------
# r_lmer(formula, data) -- lme4::lmer, REML
# ---------------------------------------------------------------------------------------------
ADAPTERS[["r_lmer"]] <- list(
  ship = FALSE,  # dense n x n fit (about 14 s at n = 300, cubic in n); the bridge keeps lme4::lmer until morie has a sparse mixed-model fitter
  native = function(a, M) {
    sp <- .rg_split_bars(a$formula[[3]])
    if (!length(sp$bars)) stop("the formula has no random-effects term ( ... | group)", call. = FALSE)
    fixed <- stats::as.formula(call("~", a$formula[[2]], sp$fixed), env = environment(a$formula))
    reml <- if (is.null(a$REML)) TRUE else isTRUE(a$REML)
    .rg_lmm_native(fixed, sp$bars, a$data, M, reml = reml)
  },
  reference = function(a) lme4::lmer(a$formula, data = a$data),
  compare = function(nat, ref) {
    list(native = c(nat$fixed$Estimate, nat$fixed$Std.Error, nat$varcomp$vcov),
         reference = c(lme4::fixef(ref), sqrt(diag(as.matrix(stats::vcov(ref)))), .rg_merMod_vc(ref)))
  },
  # The largest relative gap is the near-zero intercept-slope covariance (-0.015568 vs -0.015573,
  # absolute 5e-6, the same absolute size as the other components' gaps). lme4's REML criterion at
  # morie's estimates is 856.71036928 vs 856.71036930 at its own, i.e. morie is at least as optimal:
  # the gap is lmer's stopping tolerance along a flat direction.
  tol = 5e-4,
  args = list(formula = ymix ~ x1 + x2 + (1 + x1 | g), data = .rg_data),
  note = "same morie REML fit as r_lme (Remlik + NelderMead), lmer bar syntax parsed; varcomp = var(Int), var(x1), cov, residual as VarCorr; bobyqa tolerance ~1e-6"
)

# ---------------------------------------------------------------------------------------------
# r_glmer(formula, data, family) -- lme4::glmer, Laplace
# ---------------------------------------------------------------------------------------------
ADAPTERS[["r_glmer"]] <- list(
  ship = FALSE,  # random intercept only and no fixed-effect SEs; the bridge keeps lme4::glmer
  native = function(a, M) {
    fam <- .rg_family_name(a$family)
    if (!fam %in% c("binomial", "poisson"))
      stop("SpatialGlmmFit fits binomial or poisson GLMMs", call. = FALSE)
    sp <- .rg_split_bars(a$formula[[3]])
    if (length(sp$bars) != 1L || !identical(sp$bars[[1]][[2]], 1))
      stop("the native GLMM (SpatialGlmmFit) supports one random intercept (1 | group) only",
           call. = FALSE)
    fixed <- stats::as.formula(call("~", a$formula[[2]], sp$fixed), env = environment(a$formula))
    d <- .rg_design(fixed, a$data)
    g <- factor(eval(sp$bars[[1]][[3]], a$data, environment()))
    y <- if (fam == "binomial") .rg_binary(d$y) else as.numeric(d$y)
    r <- M$SpatialGlmmFit(y, d$X, family = fam, groups = g)
    list(method = "Laplace approximation (SpatialGlmmFit, iid group intercepts)",
         fixed = .rg_coef_table(setNames(r$beta, colnames(d$X))),
         varcomp = data.frame(group = deparse(sp$bars[[1]][[3]]), var1 = "(Intercept)",
                              vcov = r$sigma2, sdcor = sqrt(r$sigma2)),
         loglik = r$loglik, aic = r$aic, converged = r$converged)
  },
  reference = function(a) lme4::glmer(a$formula, data = a$data, family = a$family),
  compare = function(nat, ref) {
    list(native = c(nat$fixed$Estimate, nat$varcomp$vcov, nat$loglik),
         reference = c(lme4::fixef(ref), .rg_merMod_vc(ref), as.numeric(stats::logLik(ref))))
  },
  # Both maximise the Laplace likelihood; lme4 evaluates it with a PIRLS mode found to tolPwrss=1e-7,
  # morie to 1e-12. Measured on args: each objective evaluated at the other's optimum is lower by only
  # 7e-6 log-lik units (lme4 devfun: -174.1253887 at lme4 vs -174.1253957 at morie; morie's
  # Laplace: -174.1249625 at morie vs -174.1249693 at lme4), so the 1.5e-3 gap in sigma^2 is the
  # flatness of the likelihood, not a different estimator; tightening glmer's bobyqa leaves it unchanged.
  tol = 2e-3,
  args = list(formula = ybmix ~ x1 + x2 + (1 | g), data = .rg_data, family = "binomial"),
  note = "SpatialGlmmFit(groups=), same Laplace (nAGQ=1) likelihood as glmer; random intercept only, no beta SEs; sigma^2 differs ~1e-3 rel on a flat likelihood (7e-6 loglik units)"
)

# ---------------------------------------------------------------------------------------------
# r_gam(formula, data) -- mgcv::gam
# ---------------------------------------------------------------------------------------------
ADAPTERS[["r_gam"]] <- list(
  ship = FALSE,  # a different smoother (P-spline basis, not mgcv's thin plate); the bridge keeps mgcv::gam
  native = function(a, M) {
    rhs <- a$formula[[3]]
    if (!(is.call(rhs) && identical(rhs[[1]], as.name("s")) && length(rhs) == 2L))
      stop("the native GAM supports one smooth term: y ~ s(x)", call. = FALSE)
    x <- as.numeric(eval(rhs[[2]], a$data, environment()))
    y <- as.numeric(eval(a$formula[[2]], a$data, environment()))
    gcv <- function(ll) {
      r <- M$pspln(x, y, n_knots = 20L, degree = 3L, lam = exp(ll))
      M$morie_esl_gcv(y, r$fitted, r$edf)$gcv
    }
    grid <- seq(-10, 15, by = 0.5)
    ll0 <- grid[which.min(vapply(grid, gcv, 1))]
    opt <- M$NelderMead(function(p) gcv(p), ll0, xtol = 1e-9, ftol = 1e-14)
    r <- M$pspln(x, y, n_knots = 20L, degree = 3L, lam = exp(opt$x))
    list(method = "P-spline (cubic B-splines, 20 knots, 2nd-difference penalty), lambda by GCV",
         lambda = exp(opt$x), edf = r$edf, gcv = opt$fun, r2 = r$r2,
         fitted = r$fitted, coefficients = r$coef)
  },
  reference = function(a) mgcv::gam(a$formula, data = a$data),
  compare = function(nat, ref) {
    list(native = c(nat$fitted, nat$edf), reference = c(stats::fitted(ref), sum(ref$edf)))
  },
  tol = 1e-6,
  args = list(formula = ys ~ s(xs), data = .rg_data),
  mismatch = TRUE,
  note = "no native mgcv equivalent: morie has fixed-lambda smoothers (pspln, morie_esl_gam); native = pspln + GCV (morie_esl_gcv) -- same criterion as gam's GCV.Cp but a 22-dim P-spline basis vs mgcv's k=10 thin-plate basis, so fits/EDF differ by basis, not error"
)

# ---------------------------------------------------------------------------------------------
# r_quantreg(formula, data, tau) -- quantreg::rq
# ---------------------------------------------------------------------------------------------
ADAPTERS[["r_quantreg"]] <- list(
  ship = FALSE,  # dense simplex tableau (about 5 s at n = 200); the bridge keeps quantreg::rq
  native = function(a, M) {
    tau <- if (is.null(a$tau)) 0.5 else as.numeric(a$tau)
    d <- .rg_design(a$formula, a$data)
    out <- lapply(tau, function(t) {
      r <- M$QuantileRegressionLp(as.numeric(d$y), d$X, tau = t)
      setNames(r$coefficients, colnames(d$X))
    })
    cf <- do.call(cbind, out)
    colnames(cf) <- paste0("tau= ", format(tau))
    list(method = "Koenker-Bassett regression quantiles by simplex (QuantileRegressionLp)",
         coefficients = cf, tau = tau)
  },
  reference = function(a) quantreg::rq(a$formula, tau = a$tau, data = a$data),
  compare = function(nat, ref) list(native = as.numeric(nat$coefficients),
                                    reference = as.numeric(stats::coef(ref))),
  tol = 1e-8,
  args = list(formula = y ~ x1 + x2 + f, data = .rg_data[1:150, ], tau = 0.3),
  note = "QuantileRegressionLp (two-phase simplex, same LP as rq's Barrodale-Roberts); coefficients only (rq's default summary gives rank-inversion CIs, no SEs); dense tableau so O(n^2) memory"
)

# ---------------------------------------------------------------------------------------------
# r_robust(formula, data) -- MASS::rlm
# ---------------------------------------------------------------------------------------------
ADAPTERS[["r_robust"]] <- list(
  native = function(a, M) {
    fit <- M$morie_rlm(a$formula, a$data)
    s <- M$summary.morie_rlm(fit)
    list(call = "morie_rlm(formula, data)  # Huber M, k = 1.345, MAD scale",
         coefficients = as.data.frame(s$coefficients), scale = fit$s,
         converged = fit$converged, n_downweighted = fit$n_downweighted)
  },
  reference = function(a) MASS::rlm(a$formula, data = a$data),
  compare = function(nat, ref) {
    s <- summary(ref)
    list(native = c(nat$coefficients$Value, nat$coefficients$`Std. Error`, nat$scale),
         reference = c(s$coefficients[, 1], s$coefficients[, 2], ref$s))
  },
  tol = 1e-8,
  args = list(formula = yr ~ x1 + x2 + f, data = .rg_data),
  note = "morie_rlm (Huber IRLS, MAD scale, same defaults as rlm) + summary.morie_rlm SEs"
)
