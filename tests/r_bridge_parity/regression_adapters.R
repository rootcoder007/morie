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
# r_nls(formula, data, start) -- stats::nls
# ---------------------------------------------------------------------------------------------
ADAPTERS[["r_nls"]] <- list(
  native = function(a, M) {
    st <- a$start
    if (!is.list(st)) st <- as.list(st)
    fit <- M$morie_nls(a$formula, as.data.frame(a$data), st,
                       algorithm = if (is.null(a$algorithm)) "gauss-newton" else a$algorithm,
                       control = if (is.null(a$control)) list() else a$control)
    M$summary.morie_nls(fit)
  },
  reference = function(a) stats::nls(a$formula, data = a$data, start = a$start),
  compare = function(nat, ref) {
    s <- summary(ref)
    list(native = c(nat$coefficients[, 1], nat$coefficients[, 2], nat$sigma),
         reference = c(s$coefficients[, 1], s$coefficients[, 2], s$sigma))
  },
  tol = 1e-6,
  args = list(formula = ynls ~ A * (1 - exp(-k * xn)), data = .rg_data, start = list(A = 5, k = 0.2)),
  note = "morie_nls (Gauss-Newton with nls's step halving and convergence criterion; analytic derivatives from the formula)"
)

# shared by r_lme / r_lmer / r_glmer: variance components and random effects as plain tables
.lmmb_varcomp <- function(fit, residual = TRUE) {
  vt <- fit$varcorr_table
  vc <- data.frame(group = sub("\\.[0-9]+$", "", vt$grp), var1 = vt$var1, var2 = vt$var2,
                   vcov = vt$vcov, sdcor = vt$sdcor, stringsAsFactors = FALSE)
  if (residual)
    vc <- rbind(vc, data.frame(group = "Residual", var1 = NA_character_, var2 = NA_character_,
                               vcov = fit$sigma^2, sdcor = fit$sigma, stringsAsFactors = FALSE))
  rownames(vc) <- NULL
  vc
}
.lmmb_ranef <- function(fit) lapply(fit$ranef, function(r) cbind(level = rownames(r), r))

# ---------------------------------------------------------------------------------------------
# r_lme(fixed, data, random) -- nlme::lme
# ---------------------------------------------------------------------------------------------
ADAPTERS[["r_lme"]] <- list(
  native = function(a, M) {
    method <- if (is.null(a$method)) "REML" else toupper(a$method)
    reml <- identical(method, "REML")
    fit <- M$morie_lmm(a$fixed, as.data.frame(a$data), REML = reml, random = a$random)
    tab <- fit$fixef_table
    # nlme reports ML standard errors with the REML-type divisor: SE_ML * sqrt(n / (n - p))
    if (!reml) tab$Std.Error <- tab$Std.Error * sqrt(fit$nobs / (fit$nobs - nrow(tab)))
    tab[["t value"]] <- tab$Estimate / tab$Std.Error
    if (!is.null(tab$df))
      tab[["Pr(>|t|)"]] <- 2 * stats::pt(abs(tab[["t value"]]), tab$df, lower.tail = FALSE)
    names(tab)[names(tab) == "df"] <- "DF"
    list(method = sprintf("%s, nlme containment df (morie_lmm)", method),
         fixed = tab, varcomp = .lmmb_varcomp(fit), loglik = fit$logLik, aic = fit$AIC,
         bic = fit$BIC, n = fit$nobs, n_groups = fit$ngroups, ranef = .lmmb_ranef(fit),
         converged = fit$convergence$code == 0)
  },
  reference = function(a) nlme::lme(fixed = a$fixed, data = a$data, random = a$random),
  compare = function(nat, ref) {
    vc <- nlme::getVarCov(ref)
    v <- nat$varcomp
    list(native = c(nat$fixed$Estimate, nat$fixed$Std.Error, nat$fixed$DF,
                    v$vcov[is.na(v$var2) & v$group != "Residual"], v$vcov[v$group == "Residual"],
                    nat$loglik),
         reference = c(nlme::fixef(ref), sqrt(diag(stats::vcov(ref))),
                       summary(ref)$tTable[, "DF"], diag(vc), ref$sigma^2,
                       as.numeric(stats::logLik(ref))))
  },
  tol = 1e-4,
  args = list(fixed = ymix ~ x1 + x2, data = .rg_data, random = ~ 1 + x1 | g),
  note = "morie_lmm (profiled REML deviance as lme4, block-diagonal Cholesky by group level), nlme containment df; lme stops at its own optimiser tolerance (~1e-6..1e-5)"
)

# ---------------------------------------------------------------------------------------------
# r_lmer(formula, data) -- lme4::lmer
# ---------------------------------------------------------------------------------------------
ADAPTERS[["r_lmer"]] <- list(
  native = function(a, M) {
    reml <- if (is.null(a$REML)) TRUE else isTRUE(a$REML)
    fit <- M$morie_lmm(a$formula, as.data.frame(a$data), REML = reml)
    tab <- fit$fixef_table[, c("Estimate", "Std.Error", "t value")]
    list(method = sprintf("%s, profiled deviance as lme4 (morie_lmm)", if (reml) "REML" else "ML"),
         fixed = tab, varcomp = .lmmb_varcomp(fit), loglik = fit$logLik, aic = fit$AIC,
         bic = fit$BIC, reml_criterion = fit$REMLcrit, n = fit$nobs, n_groups = fit$ngroups,
         ranef = .lmmb_ranef(fit), converged = fit$convergence$code == 0)
  },
  reference = function(a) lme4::lmer(a$formula, data = a$data),
  compare = function(nat, ref) {
    list(native = c(nat$fixed$Estimate, nat$fixed$Std.Error, nat$varcomp$vcov, nat$reml_criterion),
         reference = c(lme4::fixef(ref), sqrt(diag(as.matrix(stats::vcov(ref)))), .rg_merMod_vc(ref),
                       lme4::REMLcrit(ref)))
  },
  # the near-zero intercept-slope covariance is the loosest component: lmer stops at bobyqa's
  # tolerance along a flat direction, and morie's REML criterion is never higher than lmer's
  tol = 5e-4,
  args = list(formula = ymix ~ x1 + x2 + (1 + x1 | g), data = .rg_data),
  note = "morie_lmm: lme4's profiled REML deviance (penalised least squares, block-diagonal Cholesky); varcomp as VarCorr"
)

# ---------------------------------------------------------------------------------------------
# r_glmer(formula, data, family) -- lme4::glmer, Laplace
# ---------------------------------------------------------------------------------------------
ADAPTERS[["r_glmer"]] <- list(
  native = function(a, M) {
    fam <- if (is.null(a$family)) "binomial" else .rg_family_name(a$family)
    fit <- M$morie_glmm(a$formula, as.data.frame(a$data), family = fam)
    list(method = sprintf("Laplace (nAGQ = 1), %s %s link, glmer's two-stage PIRLS (morie_glmm)",
                          fit$family$family, fit$family$link),
         fixed = fit$fixef_table, varcomp = .lmmb_varcomp(fit, residual = FALSE),
         loglik = fit$logLik, aic = fit$AIC, bic = fit$BIC, deviance = fit$deviance, n = fit$nobs,
         n_groups = fit$ngroups, ranef = .lmmb_ranef(fit), vcov_method = fit$vcov_method,
         converged = fit$convergence$code == 0)
  },
  # lme4's default PIRLS stops at tolPwrss = 1e-7 and perturbs its Laplace deviance by up to ~1e-3;
  # with tolPwrss = 1e-13 and a tight bobyqa it computes the exact Laplace deviance morie uses
  reference = function(a) lme4::glmer(a$formula, data = a$data, family = a$family,
                                      control = lme4::glmerControl(tolPwrss = 1e-13,
                                        optimizer = c("bobyqa", "bobyqa"),
                                        optCtrl = list(rhoend = 1e-12, maxfun = 1e5))),
  compare = function(nat, ref) {
    list(native = c(nat$fixed$Estimate, nat$fixed[[2]], nat$varcomp$vcov, nat$loglik),
         reference = c(lme4::fixef(ref), sqrt(diag(as.matrix(stats::vcov(ref)))), .rg_merMod_vc(ref),
                       as.numeric(stats::logLik(ref))))
  },
  tol = 1e-4,
  args = list(formula = ybmix ~ x1 + x2 + (1 | g), data = .rg_data, family = "binomial"),
  note = "morie_glmm: the exact Laplace (nAGQ = 1) deviance with glmer's two-stage PIRLS; checked against glmer run to tolPwrss = 1e-13"
)

# ---------------------------------------------------------------------------------------------
# r_gam(formula, data) -- mgcv::gam
# ---------------------------------------------------------------------------------------------
ADAPTERS[["r_gam"]] <- list(
  native = function(a, M) {
    fam <- if (is.null(a$family)) "gaussian" else .rg_family_name(a$family)
    fit <- M$morie_gam(a$formula, as.data.frame(a$data), family = fam,
                       method = if (is.null(a$method)) "GCV.Cp" else a$method,
                       weights = a$weights, sp = a$sp)
    s <- M$summary.morie_gam(fit)
    list(call = sprintf("morie_gam(formula, data, family = \"%s\", method = \"%s\")",
                        fit$family, fit$method),
         parametric = if (!is.null(s$p.table)) as.data.frame(s$p.table),
         smooth_terms = if (!is.null(s$s.table)) as.data.frame(s$s.table),
         r.sq = s$r.sq, dev.expl = s$dev.expl, score = unname(fit$score),
         method = fit$method, scale = fit$scale, sp = fit$sp,
         edf = fit$edf_smooth, edf_total = fit$edf_total, n = fit$nobs)
  },
  reference = function(a) mgcv::gam(a$formula, data = a$data),
  compare = function(nat, ref) {
    s <- summary(ref)
    # the x1 estimate itself is ~3e-5 (ys does not depend on x1): its SE is compared, not its value
    list(native = c(nat$score, nat$edf, nat$r.sq, nat$parametric[1, 1], nat$parametric[, 2]),
         reference = c(ref$gcv.ubre, s$edf, s$r.sq, s$p.table[1, 1], s$p.table[, 2]))
  },
  # mgcv stops its smoothing-parameter search at |grad| < 1e-6 x score; morie converges further,
  # so the criterion agrees to ~1e-9 and the EDF to ~1e-6
  tol = 1e-4,
  args = list(formula = ys ~ s(xs) + x1, data = .rg_data),
  note = "morie_gam: mgcv's thin plate regression spline basis and GCV/REML smoothness selection; same coefficients at mgcv's sp to ~1e-13"
)

# ---------------------------------------------------------------------------------------------
# r_quantreg(formula, data, tau) -- quantreg::rq
# ---------------------------------------------------------------------------------------------
ADAPTERS[["r_quantreg"]] <- list(
  native = function(a, M) {
    fit <- M$morie_rq(a$formula, as.data.frame(a$data),
                      tau = if (is.null(a$tau)) 0.5 else a$tau,
                      method = if (is.null(a$method)) "fn" else a$method,
                      se = if (is.null(a$se)) "nid" else a$se,
                      R = if (is.null(a$R)) 200L else a$R, seed = a$seed)
    M$summary.morie_rq(fit)
  },
  reference = function(a) quantreg::rq(a$formula, tau = a$tau, data = a$data),
  compare = function(nat, ref) {
    s <- summary(ref, se = "nid")
    tb <- nat$coefficients
    if (is.list(tb) && !is.data.frame(tb)) tb <- tb[[1]]
    list(native = c(tb[, 1], tb[, 2]),
         reference = c(as.numeric(stats::coef(ref)), s$coefficients[, 2]))
  },
  tol = 1e-6,
  args = list(formula = y ~ x1 + x2 + f, data = .rg_data, tau = 0.3),
  note = "morie_rq: Frisch-Newton interior point (as rq method \"fn\") with Barrodale-Roberts vertex polish; nid sandwich SEs as summary.rq"
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
