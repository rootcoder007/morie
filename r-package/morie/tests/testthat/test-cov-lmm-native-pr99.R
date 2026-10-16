# R/lmm_native.R, the PR #99 lines of morie_glmm the pr77 file leaves: PIRLS giving up when no step
# halving is accepted, stopping once a halved step changes nothing, a non-finite deviance read as
# Inf, a Hessian whose inverse is not positive definite, and the GLMM header of print().

.g9_data <- function(seed) {
  set.seed(seed)
  d <- data.frame(g = gl(10, 12), x = stats::rnorm(120))
  d$ys <- as.numeric(d$x > 0)                # completely separated by x
  d$yg <- as.numeric(as.integer(d$g) > 5)    # constant within each group
  d
}

.g9_quiet <- function(expr) {
  w <- character()
  val <- withCallingHandlers(expr, warning = function(x) {
    w <<- c(w, conditionMessage(x))
    invokeRestart("muffleWarning")
  })
  list(value = val, warnings = w)
}

test_that("a fit separated by x classifies every row as the data does", {
  d <- .g9_data(1)
  f <- .g9_quiet(morie_glmm(ys ~ x + (1 | g), d))$value
  # the separating direction: every fitted probability is on the side of 0.5 the response is
  expect_identical(unname(f$fitted > 0.5), d$x > 0)
  expect_gt(unname(f$coefficients[["x"]]), 0)
  expect_true(all(is.finite(f$coefficients)))
})

test_that("a response constant within groups keeps the RX covariance and says so", {
  d <- .g9_data(1)
  r <- .g9_quiet(morie_glmm(yg ~ x + (1 | g), d))
  f <- r$value
  expect_true(any(grepl("the finite-difference Hessian is not positive definite; using the RX-based covariance",
                        r$warnings, fixed = TRUE)))
  expect_null(f$vcov_hessian)
  expect_identical(f$vcov_method, "RX")
  expect_identical(f$vcov, f$vcov_RX)
  expect_identical(unname(f$fitted > 0.5), d$yg == 1)
})

test_that("the deviance function: lme4's value at the optimum, Inf for a non-finite parameter", {
  skip_if_not_installed("lme4")
  set.seed(1)
  d <- data.frame(g = gl(8, 10), x = stats::rnorm(80))
  d$y <- stats::rbinom(80, 1, stats::plogis(0.5 * d$x + stats::rnorm(8)[d$g]))
  f <- morie_glmm(y ~ x + (1 | g), d)
  ref <- lme4::glmer(y ~ x + (1 | g), d, family = stats::binomial)
  p <- c(f$theta, f$coefficients)
  expect_equal(f$devfun(p), -2 * as.numeric(stats::logLik(ref)), tolerance = 1e-5)
  expect_identical(f$devfun(replace(p, 2L, NaN)), Inf)
  # a huge random-effect scale: PIRLS stops where no halved step is accepted, at a deviance above the
  # minimum
  big <- f$devfun(replace(p, 1L, 1e6))
  expect_true(is.finite(big))
  expect_gt(big, f$devfun(p))
})

test_that("print names the GLMM's family and link", {
  set.seed(2)
  d <- data.frame(g = gl(6, 10), x = stats::rnorm(60))
  d$y <- stats::rpois(60, exp(0.3 + 0.4 * d$x))
  f <- .g9_quiet(morie_glmm(y ~ x + (1 | g), d, family = "poisson"))$value
  out <- utils::capture.output(res <- print(f))
  expect_identical(out[1L], "Generalized linear mixed model fit by maximum likelihood (Laplace) [poisson, log link]")
  expect_identical(out[2L], "Formula: y ~ x + (1 | g) ")
  expect_identical(res, f)
})
