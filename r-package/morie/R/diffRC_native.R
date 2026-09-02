# DiffRec: diffusion over interactions, with the noise turned down.
# Sources: Wang, W., Xu, Y., Feng, F., Lin, X., He, X. & Chua, T.-S.
# (2023) "Diffusion Recommender Model", SIGIR '23, 832-841
# (arXiv:2304.04971) -- the criticism of GAN/VAE recommenders, the
# diffusion formulation over interaction histories, the reduced noise
# scale, and importance sampling over timesteps; Ho, J., Jain, A. &
# Abbeel, P. (2020) "Denoising Diffusion Probabilistic Models",
# NeurIPS 2020 (arXiv:2006.11239) -- the forward process and the
# closed form used here; Liang, D. et al. (2018) "Variational
# Autoencoders for Collaborative Filtering", WWW 2018 (the VAE
# recommender being displaced).
#
# Native implementation mirroring morie.fn.diffRC exactly: same
# scaled linear beta schedule, same closed-form forward corruption
# (no simulation needed to check it), same DDPM posterior mean, same
# importance weights on the square-root losses with a smoothing
# constant, same reverse chain that must be the identity at scale 0.

.ghc_DIFFRC_EPS <- 1e-12

#' Linear noise schedule, scaled down
#'
#' At image-diffusion scales the personal history is destroyed. The
#' \code{scale} knob keeps it recoverable.
#'
#' @param T Number of diffusion steps.
#' @param scale Noise scale multiplier.
#' @param beta_min Lower endpoint of the unscaled linear schedule.
#' @param beta_max Upper endpoint of the unscaled linear schedule.
#' @return A list with \code{beta}, \code{alpha_bar}, \code{T},
#'   \code{scale}, \code{signal_retained}, \code{note}.
#' @export
noise_schedule <- function(T, scale = 0.001, beta_min = 0.0001,
                           beta_max = 0.02) {
  n <- as.integer(T)
  s <- as.numeric(scale)
  if (n < 1L) stop("diffRC: T must be at least 1")
  if (s < 0) stop("diffRC: the noise scale cannot be negative")
  denom <- max(n - 1L, 1L)
  betas <- s * (beta_min + (beta_max - beta_min) * ((seq_len(n) - 1L) / denom))
  abar <- cumprod(1 - betas)
  list(beta = as.numeric(betas), alpha_bar = as.numeric(abar),
       T = n, scale = s, signal_retained = abar[n],
       note = paste("scale = 0 leaves alpha_bar at 1, so the forward",
                    "process is the identity"))
}

#' Forward corruption (closed form)
#'
#' Mean \code{sqrt(alpha_bar_t) * x_0}, variance \code{1 - alpha_bar_t}.
#' No simulation is needed to check it.
#'
#' @param x0 Original history.
#' @param alpha_bar_t \code{alpha_bar} at time \code{t}.
#' @param e Shared generator environment. If \code{NULL} or
#'   \code{std == 0}, no sample is drawn.
#' @return A list with \code{x_t}, \code{mean}, \code{std},
#'   \code{sampled}.
#' @export
forward_corrupt <- function(x0, alpha_bar_t, e = NULL) {
  x <- as.numeric(x0)
  ab <- as.numeric(alpha_bar_t)
  if (!(0 <= ab && ab <= 1))
    stop(sprintf("diffRC: alpha_bar must lie in [0,1], got %g", ab))
  sm <- sqrt(ab)
  sv <- sqrt(max(1 - ab, 0))
  mean_v <- sm * x
  if (is.null(e) || sv <= .ghc_DIFFRC_EPS)
    return(list(x_t = mean_v, mean = mean_v, std = sv, sampled = FALSE))
  u <- .ghc_unif(e, length(x))
  xt <- mean_v + sv * (2 * u - 1) * sqrt(3)
  list(x_t = xt, mean = mean_v, std = sv, sampled = TRUE)
}

#' DDPM posterior mean
#'
#' \code{mu(x_t, x_0_hat)} with the standard coefficients.
#'
#' @param x_t Current noisy state.
#' @param x0_hat Model estimate of \code{x_0}.
#' @param alpha_bar_t \code{alpha_bar} at \code{t}.
#' @param alpha_bar_prev \code{alpha_bar} at \code{t-1}.
#' @param beta_t \code{beta} at \code{t}.
#' @return A list with \code{mean}, \code{coef_x0}, \code{coef_xt},
#'   \code{degenerate}.
#' @export
posterior_mean <- function(x_t, x0_hat, alpha_bar_t, alpha_bar_prev,
                            beta_t) {
  xt <- as.numeric(x_t); x0 <- as.numeric(x0_hat)
  if (length(xt) != length(x0))
    stop("diffRC: x_t and the estimate of x_0 differ in length")
  ab <- as.numeric(alpha_bar_t)
  abp <- as.numeric(alpha_bar_prev)
  b <- as.numeric(beta_t)
  denom <- 1 - ab
  if (denom <= .ghc_DIFFRC_EPS)
    return(list(mean = x0, degenerate = TRUE,
                note = "alpha_bar = 1: nothing was added, so nothing is removed"))
  c0 <- sqrt(max(abp, 0)) * b / denom
  ct <- sqrt(max(1 - b, 0)) * (1 - abp) / denom
  list(mean = c0 * x0 + ct * xt, coef_x0 = c0, coef_xt = ct,
       degenerate = FALSE)
}

#' Importance weights over diffusion steps
#'
#' Sample the steps that actually carry loss instead of drawing
#' uniformly.
#'
#' @param step_losses Numeric per-step losses.
#' @param uniform Use uniform weights.
#' @param smoothing Additive constant in the square-root.
#' @return A list with \code{weights}, \code{uniform},
#'   \code{effective_steps} (and \code{note} for the importance path).
#' @export
importance_weights <- function(step_losses, uniform = FALSE,
                                smoothing = 0.1) {
  L <- as.numeric(step_losses)
  if (length(L) == 0L) stop("diffRC: no per-step losses given")
  if (any(L < 0)) stop("diffRC: a loss cannot be negative")
  n <- length(L)
  if (isTRUE(uniform))
    return(list(weights = rep(1 / n, n), uniform = TRUE,
                effective_steps = as.numeric(n)))
  s <- as.numeric(smoothing)
  w <- sqrt(L) + s
  z <- sum(w)
  p <- w / z
  eff <- 1 / sum(p^2)
  list(weights = as.numeric(p), uniform = FALSE,
       effective_steps = eff,
       note = paste("effective sample size falls as the loss",
                    "concentrates, which is the intended behaviour"))
}

#' Reverse chain
#'
#' Walks the schedule from \code{t} back to 0. With \code{scale = 0}
#' the schedule is the identity and this returns its input unchanged.
#'
#' @param x_t Noisy state to start from.
#' @param model Function \code{(x, t)} returning the model's
#'   \code{x_0} estimate.
#' @param schedule Schedule list from \code{noise_schedule}.
#' @param t_start Starting timestep; defaults to the last step.
#' @return A list mirroring the Python \code{RichResult} payload.
#' @export
denoise <- function(x_t, model, schedule, t_start = NULL) {
  x <- as.numeric(x_t)
  ab <- schedule$alpha_bar
  beta <- schedule$beta
  T <- as.integer(schedule$T)
  t <- if (is.null(t_start)) T - 1L else as.integer(t_start)
  if (t < 0L || t >= T) stop("diffRC: t is outside the schedule")
  path <- list()
  while (t >= 0L) {
    x0h <- as.numeric(model(x, t))
    prev <- if (t > 0L) ab[t] else 1
    pm <- posterior_mean(x, x0h, ab[t + 1L], prev, beta[t + 1L])
    x <- pm$mean
    path[[length(path) + 1L]] <- x
    t <- t - 1L
  }
  list(estimate = x, x0 = x, path = path, steps = length(path),
       signal_retained = schedule$signal_retained,
       method = "diffusion over interaction histories; Wang et al. (2023)",
       note = paste("the noise scale is REDUCED so the personalised",
                    "history survives the forward process"))
}

#' Cheat sheet for the diffRC module
#'
#' One-screen reminder of the module's entry points, printed to the console.
#'
#' @return The cheat sheet text, invisibly.
#' @export
.diffRC_cheatsheet <- function() {
  paste("diffRC: GAN recommenders are unstable and VAE ones trade",
        "representation for tractability, so use diffusion -- but",
        "NOT the image schedule. Image diffusion destroys x_0",
        "because sampling starts from noise; a user's interaction",
        "history is the personalised signal being predicted, so",
        "corrupting it fully erases the target. DiffRec adds noise",
        "on a REDUCED scale. Forward is closed form: mean",
        "sqrt(alpha_bar) x_0, variance 1 - alpha_bar. At scale 0",
        "the whole thing must be the IDENTITY. Timesteps are drawn",
        "by IMPORTANCE SAMPLING, so the steps that carry loss are",
        "not visited by luck.")
}

# compact alias per ledger/NAMING.md
#' @export
diffusionrecommender <- denoise

# public names resolved by fn/_lazy_map.json
#' @export
diffusion_rec <- denoise
#' @export
diffusionrec <- denoise

# house entry point: the package exports one morie_<module>
morie_diffRC <- denoise
