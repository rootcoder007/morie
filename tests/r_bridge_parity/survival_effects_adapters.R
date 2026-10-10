# Adapters for GROUP survival_effects: r_surv r_survfit r_coxph r_survdiff r_aft r_p_adjust
# r_vif r_durbinwatson r_bptest r_resettest r_cohens_d r_hedges_g r_eta_sq r_cramers_v
# r_power_t r_power_anova r_power_chisq r_power_prop
if (!exists("ADAPTERS")) ADAPTERS <- list()

# ---- helpers (base R only) -------------------------------------------------------------------

# Split a `Surv(time, event) ~ rhs` formula into time, event (0/1) and the RHS pieces, without
# calling survival::Surv: model.frame reads the two arguments of the Surv() call from `data`, as
# it reads any formula variable.
.se_lhs_value <- function(expr, data, env) {
  stats::model.frame(stats::as.formula(call("~", expr), env = env), data, na.action = stats::na.pass)[[1]]
}
.se_surv_parts <- function(f, data) {
  lhs <- f[[2]]
  if (!is.call(lhs) || !identical(as.character(lhs[[1]]), "Surv") &&
      !identical(deparse(lhs[[1]]), "survival::Surv"))
    stop("left-hand side must be Surv(time, event)")
  if (length(lhs) != 3L) stop("only right-censored Surv(time, event) is supported")
  env <- environment(f) %||% parent.frame()
  time <- as.numeric(.se_lhs_value(lhs[[2]], data, env))
  ev <- .se_lhs_value(lhs[[3]], data, env)
  ev <- as.numeric(ev)
  if (all(ev %in% c(1, 2)) && any(ev == 2)) ev <- ev - 1   # Surv's 1/2 coding
  rt <- stats::delete.response(stats::terms(f))
  mf <- stats::model.frame(rt, data)
  X <- stats::model.matrix(rt, mf)
  X <- X[, colnames(X) != "(Intercept)", drop = FALSE]
  grp <- if (ncol(mf) == 0L) factor(rep("all", length(time))) else {
    labs <- lapply(names(mf), function(v) paste0(v, "=", mf[[v]]))
    lv <- lapply(names(mf), function(v) {
      x <- mf[[v]]; paste0(v, "=", if (is.factor(x)) levels(x) else sort(unique(x)))
    })
    if (length(labs) == 1L) factor(labs[[1]], levels = lv[[1]])
    else interaction(lapply(seq_along(labs), function(i) factor(labs[[i]], levels = lv[[i]])),
                     sep = ", ", drop = TRUE, lex.order = TRUE)
  }
  list(time = time, event = ev, X = X, group = droplevels(grp), strata = ncol(mf) > 0L)
}
`%||%` <- function(a, b) if (is.null(a)) b else a

.se_yX <- function(f, data, intercept = FALSE) {
  mf <- stats::model.frame(f, data)
  y <- as.numeric(stats::model.response(mf))
  X <- stats::model.matrix(stats::terms(mf), mf)
  if (!intercept) X <- X[, colnames(X) != "(Intercept)", drop = FALSE]
  list(y = y, X = X)
}

# The bridge attaches survival before calling it, so `Surv` in a formula resolves; mirror that
# for the reference calls only (natives never touch survival).
.se_sf <- function(f) {
  environment(f) <- list2env(list(Surv = survival::Surv), parent = environment(f) %||% globalenv())
  f
}

.se_reldiff <- function(a, b) {
  a <- as.numeric(a); b <- as.numeric(b)
  max(abs(a - b) / pmax(abs(b), 1e-12))
}

# ---- simulated data ----------------------------------------------------------------------------
set.seed(20261010)
.se_n <- 300
.se_surv_df <- local({
  x1 <- rnorm(.se_n); x2 <- rbinom(.se_n, 1, 0.4)
  grp <- factor(sample(c("A", "B", "C"), .se_n, replace = TRUE))
  lin <- 0.5 * x1 - 0.7 * x2 + c(A = 0, B = 0.4, C = -0.3)[as.character(grp)]
  t_true <- rweibull(.se_n, shape = 1.4, scale = exp(1.5 - lin / 1.4))
  cens <- runif(.se_n, 0, 12)
  # round to 1 decimal so tied event times occur (exercises Efron / Greenwood ties)
  time <- pmax(round(pmin(t_true, cens), 1), 0.1)
  data.frame(time = time, status = as.integer(t_true <= cens), x1 = x1, x2 = x2, grp = grp)
})
.se_lm_df <- local({
  n <- 250
  x1 <- rnorm(n); x2 <- 0.6 * x1 + rnorm(n, sd = 0.8); x3 <- runif(n, -2, 2)
  e <- as.numeric(stats::filter(rnorm(n, sd = 1 + 0.4 * abs(x3)), 0.3, method = "recursive"))
  y <- 1 + 0.8 * x1 - 0.5 * x2 + 0.4 * x3 + 0.15 * x3^2 + e
  data.frame(y = y, x1 = x1, x2 = x2, x3 = x3)
})
.se_aov_df <- local({
  n <- 240
  a <- factor(sample(c("lo", "mid", "hi"), n, TRUE)); b <- factor(sample(c("u", "v"), n, TRUE))
  y <- c(lo = 0, mid = 0.5, hi = 1)[as.character(a)] + 0.3 * (b == "v") + rnorm(n)
  data.frame(y = y, a = a, b = b)
})
.se_x <- rnorm(80, 10.6, 2); .se_y <- rnorm(95, 10, 2.3)
.se_cx <- factor(sample(c("r", "s", "t"), 400, TRUE, prob = c(.5, .3, .2)))
.se_cy <- factor(ifelse(runif(400) < c(r = .3, s = .5, t = .6)[as.character(.se_cx)], "yes",
                        sample(c("no", "maybe"), 400, TRUE)))
.se_p <- c(0.0002, 0.003, 0.004, 0.012, 0.019, 0.03, 0.047, 0.08, 0.2, 0.33, 0.51, 0.74, 0.9)

# ---- survival ------------------------------------------------------------------------------

ADAPTERS[["r_surv"]] <- list(
  native = NULL,
  reference = function(a) survival::Surv(a$time, a$event),
  compare = function(nat, ref) list(native = numeric(0), reference = numeric(0)),
  tol = 0,
  args = list(time = .se_surv_df$time, event = .se_surv_df$status),
  note = paste("No native: survival::Surv only builds a 'Surv' response object (time+status",
               "matrix with a print method); morie's survival natives take time and event",
               "vectors directly, so there is no object constructor to map to.")
)

ADAPTERS[["r_survfit"]] <- list(
  native = function(a, M) {
    s <- .se_surv_parts(a$formula, a$data)
    out <- lapply(levels(s$group), function(g) {
      i <- s$group == g
      k <- M$morie_kaplan_meier(s$time[i], s$event[i])
      data.frame(strata = if (s$strata) g else NA_character_, time = k$time,
                 n.risk = k$n_risk, n.event = k$n_event, surv = k$surv, std.err = k$se,
                 lower = k$lower, upper = k$upper, stringsAsFactors = FALSE)
    })
    out <- do.call(rbind, out)
    if (!s$strata) out$strata <- NULL
    rownames(out) <- NULL
    out
  },
  reference = function(a) survival::survfit(.se_sf(a$formula), data = a$data),
  compare = function(nat, ref) {
    sm <- summary(ref)   # event times only, std.err on the S(t) scale
    # where S(t) drops to 0 survfit reports std.err NaN (Greenwood undefined), morie 0: skip those
    ok <- is.finite(sm$std.err)
    list(native = c(nat$time, nat$n.risk, nat$n.event, nat$surv, nat$std.err[ok]),
         reference = c(sm$time, sm$n.risk, sm$n.event, sm$surv, sm$std.err[ok]))
  },
  tol = 1e-10,
  args = list(formula = Surv(time, status) ~ grp, data = .se_surv_df),
  note = paste("morie_kaplan_meier per stratum; times, n.risk, n.event, S(t), Greenwood SE of S",
               "match summary(survfit). CI not compared: morie uses log-log, survfit's default",
               "conf.type is 'log'. Where S(t) hits 0 morie's SE is 0, survfit's NaN (skipped).")
)

ADAPTERS[["r_coxph"]] <- list(
  native = function(a, M) {
    s <- .se_surv_parts(a$formula, a$data)
    r <- M$morie_cox_ph(s$time, s$event, s$X, ties = a$ties %||% "efron")
    names(r$coef) <- names(r$se) <- names(r$z) <- names(r$p_value) <- names(r$hazard_ratio) <-
      colnames(s$X)
    r$table <- data.frame(coef = r$coef, `exp(coef)` = r$hazard_ratio, `se(coef)` = r$se,
                          z = r$z, p = r$p_value, check.names = FALSE)
    r
  },
  reference = function(a) survival::coxph(.se_sf(a$formula), data = a$data, ties = a$ties %||% "efron"),
  compare = function(nat, ref) list(
    native = c(nat$coef, nat$se, nat$loglik, nat$loglik_null),
    reference = c(stats::coef(ref), sqrt(diag(stats::vcov(ref))), ref$loglik[2], ref$loglik[1])),
  tol = 1e-7,
  args = list(formula = Surv(time, status) ~ x1 + x2 + grp, data = .se_surv_df),
  note = "morie_cox_ph (Newton-Raphson, Efron ties as coxph's default); coef, SE, logliks match."
)

ADAPTERS[["r_survdiff"]] <- list(
  native = function(a, M) {
    s <- .se_surv_parts(a$formula, a$data)
    M$morie_logrank_test(s$time, s$event, s$group)
  },
  reference = function(a) survival::survdiff(.se_sf(a$formula), data = a$data),
  compare = function(nat, ref) list(
    native = c(nat$statistic, nat$df, nat$observed, nat$expected),
    reference = c(ref$chisq, length(ref$n) - 1, ref$obs, ref$exp)),
  tol = 1e-10,
  args = list(formula = Surv(time, status) ~ grp, data = .se_surv_df),
  note = "morie_logrank_test (survdiff rho = 0); chisq, df, observed, expected match."
)

ADAPTERS[["r_aft"]] <- list(
  native = function(a, M) {
    s <- .se_surv_parts(a$formula, a$data)
    M$morie_aft(s$time, s$event, if (ncol(s$X)) s$X else NULL, dist = a$dist %||% "weibull")
  },
  reference = function(a) survival::survreg(.se_sf(a$formula), data = a$data, dist = a$dist %||% "weibull"),
  compare = function(nat, ref) list(
    native = c(nat$coefficients, nat$scale, nat$loglik, sqrt(diag(nat$vcov))),
    reference = c(stats::coef(ref), ref$scale, ref$loglik[2], sqrt(diag(stats::vcov(ref))))),
  tol = 1e-6,
  args = list(formula = Surv(time, status) ~ x1 + x2 + grp, data = .se_surv_df,
              dist = "weibull"),
  note = paste("morie_aft (ML, BFGS + Newton polish); dists weibull/lognormal/loglogistic/",
               "exponential only (survreg's gaussian/logistic/t on untransformed time not native).")
)

# ---- multiple testing ----------------------------------------------------------------------

ADAPTERS[["r_p_adjust"]] <- list(
  native = function(a, M) {
    m <- a$method %||% "holm"
    if (identical(m, "none")) return(as.numeric(a$p))
    key <- switch(m, BH = , fdr = "bh", BY = "by", tolower(m))
    M$adjust_p_values(a$p, method = key)$adjusted
  },
  reference = function(a) stats::p.adjust(a$p, method = a$method %||% "holm"),
  compare = function(nat, ref) list(native = nat, reference = ref),
  tol = 1e-12,
  args = list(p = .se_p, method = "BH"),
  note = paste("adjust_p_values (holm/hochberg/hommel/bonferroni/BH/fdr/BY; 'none' passes p).",
               "Its FWER/FDR arms are thin wrappers over stats::p.adjust (base R, not an outside",
               "package), so parity is exact by construction.")
)

# ---- regression diagnostics ----------------------------------------------------------------

ADAPTERS[["r_vif"]] <- list(
  native = function(a, M) {
    d <- .se_yX(a$formula, a$data)
    M$compute_vif(d$X, colnames(d$X))
  },
  reference = function(a) car::vif(stats::lm(a$formula, data = a$data)),
  compare = function(nat, ref) list(native = nat, reference = ref),
  tol = 1e-8,
  args = list(formula = y ~ x1 + x2 + x3, data = .se_lm_df),
  note = paste("compute_vif on the model matrix (intercept dropped). Numeric predictors only:",
               "for a factor with >2 levels car reports GVIF per term, morie a VIF per dummy.")
)

ADAPTERS[["r_durbinwatson"]] <- list(
  native = function(a, M) {
    d <- .se_yX(a$formula, a$data)
    M$morie_durbin_watson(M$morie_ols(d$y, d$X, add_intercept = TRUE))
  },
  reference = function(a) car::durbinWatsonTest(stats::lm(a$formula, data = a$data)),
  compare = function(nat, ref) list(native = c(nat$statistic, nat$rho),
                                    reference = c(ref$dw, ref$r)),
  tol = 1e-10,
  args = list(formula = y ~ x1 + x2 + x3, data = .se_lm_df),
  note = paste("morie_durbin_watson(morie_ols); DW statistic and lag-1 autocorrelation match.",
               "morie returns no p-value (no exact/analytic one either), car's is a bootstrap",
               "p-value, so only the statistic is compared; lmtest::dwtest not applicable.")
)

ADAPTERS[["r_bptest"]] <- list(
  native = function(a, M) {
    d <- .se_yX(a$formula, a$data)
    M$morie_breusch_pagan(M$morie_ols(d$y, d$X, add_intercept = TRUE))
  },
  reference = function(a) lmtest::bptest(stats::lm(a$formula, data = a$data)),
  compare = function(nat, ref) list(native = c(nat$statistic, nat$df, nat$p_value),
                                    reference = c(ref$statistic, ref$parameter, ref$p.value)),
  tol = 1e-8,
  args = list(formula = y ~ x1 + x2 + x3, data = .se_lm_df),
  note = "morie_breusch_pagan (Koenker studentized n*R^2 = bptest default studentize = TRUE)."
)

ADAPTERS[["r_resettest"]] <- list(
  native = function(a, M) {
    d <- .se_yX(a$formula, a$data, intercept = TRUE)
    M$ramsey_reset_test(d$y, d$X, powers = a$power %||% 2:3)
  },
  reference = function(a) lmtest::resettest(stats::lm(a$formula, data = a$data),
                                            power = a$power %||% 2:3, type = "fitted"),
  compare = function(nat, ref) list(native = c(nat$statistic, nat$df, nat$p_value),
                                    reference = c(ref$statistic, ref$parameter[1], ref$p.value)),
  tol = 1e-8,
  args = list(formula = y ~ x1 + x2 + x3, data = .se_lm_df),
  note = paste("ramsey_reset_test(y, X incl. intercept, powers = 2:3) = resettest default",
               "(power = 2:3, type = 'fitted'); powers passed explicitly.")
)

# ---- effect sizes --------------------------------------------------------------------------

ADAPTERS[["r_cohens_d"]] <- list(
  native = function(a, M) M$cohens_d(a$x, a$y),
  reference = function(a) effectsize::cohens_d(a$x, a$y),
  compare = function(nat, ref) list(native = nat$estimate, reference = ref$Cohens_d),
  tol = 1e-12,
  args = list(x = .se_x, y = .se_y),
  note = paste("cohens_d (pooled SD); estimate matches. CI not compared: morie uses the",
               "Hedges-Olkin normal SE, effectsize a noncentral-t interval.")
)

ADAPTERS[["r_hedges_g"]] <- list(
  native = function(a, M) M$hedges_g(a$x, a$y),
  reference = function(a) effectsize::hedges_g(a$x, a$y),
  compare = function(nat, ref) list(native = nat$estimate, reference = ref$Hedges_g),
  tol = 1e-10,
  args = list(x = .se_x, y = .se_y),
  note = "hedges_g (exact gamma J correction, as effectsize); estimate matches; CI method differs."
)

ADAPTERS[["r_eta_sq"]] <- list(
  native = function(a, M) {
    # Type I (sequential) sums of squares, as effectsize reads them from an aov fit; partial
    # eta^2 per term = SS_term / (SS_term + SS_resid), effectsize's default partial = TRUE.
    tab <- stats::anova(stats::lm(a$formula, data = a$data))
    rn <- trimws(rownames(tab))
    ss_res <- tab[["Sum Sq"]][rn == "Residuals"]
    terms <- rn[rn != "Residuals"]
    partial <- a$partial %||% TRUE
    ss_tot <- sum(tab[["Sum Sq"]])
    est <- vapply(terms, function(t) {
      ss <- tab[["Sum Sq"]][rn == t]
      M$eta_squared(ss, if (partial) ss + ss_res else ss_tot)$estimate
    }, numeric(1))
    data.frame(Parameter = terms, Eta2 = unname(est), partial = partial)
  },
  reference = function(a) effectsize::eta_squared(stats::aov(a$formula, data = a$data),
                                                  partial = a$partial %||% TRUE),
  compare = function(nat, ref) list(
    native = nat$Eta2,
    reference = if (!is.null(ref$Eta2_partial)) ref$Eta2_partial else ref$Eta2),
  tol = 1e-10,
  args = list(formula = y ~ a + b, data = .se_aov_df),
  note = paste("eta_squared(ss_effect, ss_total) per term from sequential SS (stats::anova);",
               "partial by default like effectsize; CI not computed natively.")
)

ADAPTERS[["r_cramers_v"]] <- list(
  native = function(a, M) {
    r <- M$cramers_v(table(a$x, a$y))
    # effectsize's default adjust = TRUE is Bergsma's bias correction, which morie returns as
    # extra$bias_corrected_v; report that as the headline unless adjust = FALSE.
    if (isTRUE(a$adjust %||% TRUE)) {
      r$extra$unadjusted_v <- r$estimate
      r$estimate <- r$extra$bias_corrected_v
      r$ci_lower <- r$ci_upper <- NA_real_
      r$measure <- "Cramer's V (adjusted)"
    }
    r
  },
  reference = function(a) {
    r <- effectsize::cramers_v(a$x, a$y, adjust = a$adjust %||% TRUE)
    # the unadjusted V too, so both of morie's numbers are checked
    attr(r, "unadjusted") <- effectsize::cramers_v(a$x, a$y, adjust = FALSE)$Cramers_v
    r
  },
  compare = function(nat, ref) list(
    native = c(nat$estimate, nat$extra$unadjusted_v %||% nat$estimate),
    reference = c(ref$Cramers_v_adjusted %||% ref$Cramers_v, attr(ref, "unadjusted"))),
  tol = 1e-10,
  args = list(x = .se_cx, y = .se_cy),
  note = paste("cramers_v: default (adjust = TRUE) mapped to morie's Bergsma bias_corrected_v,",
               "plain V checked against adjust = FALSE. CIs differ (effectsize one-sided).")
)

# ---- power ---------------------------------------------------------------------------------

ADAPTERS[["r_power_t"]] <- list(
  native = function(a, M) {
    alt <- switch(a$alternative %||% "two.sided", two.sided = "two-sided", "one-sided")
    if (!is.null(a$alternative) && a$alternative == "less") stop("alternative 'less' not mapped")
    typ <- switch(a$type %||% "two.sample", two.sample = "two-sample", one.sample = "one-sample",
                  paired = "paired")
    sol <- M$PowerTTest(n = a$n, delta = a$d, sd = 1, alpha = a$sig.level %||% 0.05,
                        power = a$power, alternative = alt, type = typ, strict = TRUE)
    out <- list(n = a$n, d = a$d, sig.level = a$sig.level %||% 0.05, power = a$power,
                alternative = a$alternative %||% "two.sided", type = typ)
    if (is.null(a$n)) out$n <- sol else if (is.null(a$power)) out$power <- sol else out$d <- sol
    out
  },
  reference = function(a) pwr::pwr.t.test(n = a$n, d = a$d, sig.level = a$sig.level %||% 0.05,
                                          power = a$power),
  compare = function(nat, ref) list(native = nat$n, reference = ref$n),
  tol = 1e-6,
  args = list(d = 0.45, n = NULL, sig.level = 0.05, power = 0.85),
  note = paste("PowerTTest(delta = d, sd = 1, strict = TRUE: both rejection tails like pwr);",
               "solves n, power or d. (morie_power_t_test wraps stats::power.t.test, one tail.)")
)

ADAPTERS[["r_power_anova"]] <- list(
  native = function(a, M) {
    sol <- M$PowerAnova(n = a$n, k = a$k, f = a$f, alpha = a$sig.level %||% 0.05, power = a$power)
    out <- list(k = a$k, n = a$n, f = a$f, sig.level = a$sig.level %||% 0.05, power = a$power)
    if (is.null(a$power)) out$power <- sol else if (is.null(a$n)) out$n <- sol
    else if (is.null(a$f)) out$f <- sol else out$k <- sol
    out
  },
  reference = function(a) pwr::pwr.anova.test(k = a$k, n = a$n, f = a$f,
                                              sig.level = a$sig.level %||% 0.05, power = a$power),
  compare = function(nat, ref) list(native = c(nat$n, nat$power), reference = c(ref$n, ref$power)),
  tol = 1e-6,
  args = list(k = 4, n = NULL, f = 0.25, sig.level = 0.05, power = 0.8),
  note = "PowerAnova (noncentral F, ncp = k n f^2, as pwr.anova.test); solves n/power/f/k."
)

ADAPTERS[["r_power_chisq"]] <- list(
  # morie has no chi-square power function; this is the noncentral chi-square power in base R
  # (stats::pchisq / qchisq / uniroot), solving for whichever of w, N, power, sig.level is NULL
  native = function(a, M) {
    sig <- if ("sig.level" %in% names(a)) a$sig.level else 0.05
    pw <- function(w, N, df, sig) {
      stats::pchisq(stats::qchisq(sig, df, lower.tail = FALSE), df, ncp = N * w^2, lower.tail = FALSE)
    }
    w <- a$w; N <- a$N; df <- a$df; power <- a$power
    if (is.null(df)) stop("df is required", call. = FALSE)
    unknown <- c(w = is.null(w), N = is.null(N), power = is.null(power), sig.level = is.null(sig))
    if (sum(unknown) != 1L) stop("leave exactly one of w, N, power, sig.level out (NULL)", call. = FALSE)
    solve <- function(f, lo, hi) stats::uniroot(f, c(lo, hi), tol = 1e-10, extendInt = "upX")$root
    if (unknown[["power"]]) power <- pw(w, N, df, sig)
    else if (unknown[["N"]]) N <- solve(function(n) pw(w, n, df, sig) - power, 2 + 1e-10, 1e7)
    else if (unknown[["w"]]) w <- solve(function(x) pw(x, N, df, sig) - power, 1e-10, 1e5)
    else sig <- stats::uniroot(function(s) pw(w, N, df, s) - power, c(1e-10, 1 - 1e-10), tol = 1e-10)$root
    structure(list(w = w, N = N, df = df, sig.level = sig, power = power,
                   method = "Chi squared power calculation", note = "N is the number of observations"),
              class = "power.htest")
  },
  reference = function(a) pwr::pwr.chisq.test(w = a$w, N = a$N, df = a$df,
                                              sig.level = a$sig.level %||% 0.05, power = a$power),
  compare = function(nat, ref) list(native = c(w = nat$w, N = nat$N, power = nat$power, sig = nat$sig.level),
                                    reference = c(w = ref$w, N = ref$N, power = ref$power, sig = ref$sig.level)),
  tol = 1e-6,
  args = list(w = 0.3, N = 120, df = 3, sig.level = 0.05),
  note = "base-R noncentral chi-square power (pchisq ncp = N w^2), as pwr.chisq.test; solves power, N, w or sig.level"
)

ADAPTERS[["r_power_prop"]] <- list(
  native = function(a, M) {
    # pwr.2p.test takes Cohen's h; PowerPropTest(method = "cohen_h") takes p1, p2 and computes
    # h = |2 asin(sqrt p1) - 2 asin(sqrt p2)|. Choose p1, p2 symmetric about 1/2 with that h.
    h <- abs(a$h)
    if (h > pi) stop("|h| must be <= pi")
    p1 <- sin((pi / 2 + h / 2) / 2)^2
    p2 <- sin((pi / 2 - h / 2) / 2)^2
    alt <- switch(a$alternative %||% "two.sided", two.sided = "two-sided", "one-sided")
    sol <- M$PowerPropTest(n = a$n, p1 = p1, p2 = p2, alpha = a$sig.level %||% 0.05,
                           power = a$power, alternative = alt, method = "cohen_h")
    out <- list(h = a$h, n = a$n, sig.level = a$sig.level %||% 0.05, power = a$power,
                alternative = a$alternative %||% "two.sided")
    if (is.null(a$power)) out$power <- sol else out$n <- sol
    out
  },
  reference = function(a) pwr::pwr.2p.test(h = a$h, n = a$n, sig.level = a$sig.level %||% 0.05,
                                           power = a$power),
  compare = function(nat, ref) list(native = c(nat$n, nat$power), reference = c(ref$n, ref$power)),
  tol = 1e-6,
  args = list(h = 0.3, n = 150, sig.level = 0.05),
  note = paste("PowerPropTest(method = 'cohen_h') with p1, p2 chosen to give the requested h;",
               "solves power or n (not h).")
)
