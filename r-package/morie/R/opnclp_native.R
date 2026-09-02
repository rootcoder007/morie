# morie.fn -- function file (rootcoder007/morie)
# OpenCLIP: scaling laws you can actually reproduce.
#
# Sources (from the Python docstring's References section):
# - Cherti, M., Beaumont, R., Wightman, R., Wortsman, M., Ilharco, G.,
#   Gordon, C., Schuhmann, C., Schmidt, L. & Jitsev, J. (2023)
#   "Reproducible scaling laws for contrastive language-image
#   learning", CVPR 2023, 2818-2829, arXiv:2212.07143.
# - Radford, A. et al. (2021) "Learning Transferable Visual Models
#   From Natural Language Supervision", ICML 2021, PMLR 139, 8748-8763,
#   arXiv:2103.00020.
# - Kaplan, J. et al. (2020) "Scaling Laws for Neural Language Models",
#   arXiv:2001.08361.

#' morie_opnclp
#'
#' A step of the opnclp_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param payload A list; the body reads \code{$compute}, \code{$fit}, \code{$image_embeddings}, \code{$label_a}, \code{$label_b}, \code{$model_params}, \code{$op}, \code{$samples_seen}, \code{$temperature}, \code{$text_embeddings}, \code{$x}, \code{$x_a}, \code{$x_b}, \code{$y}, \code{$y_a}, \code{$y_b} from it.
#' @return Nothing; this branch always raises.
#' @export
morie_opnclp <- function(payload) {
  if (!is.list(payload) || is.null(payload$op)) {
    stop("opnclp: payload must be a list with an 'op' field")
  }
  op <- as.character(payload$op)
  if (op == "total_compute") {
    return(total_compute(payload$samples_seen, payload$model_params))
  }
  if (op == "fit_power_law") {
    return(fit_power_law(payload$x, payload$y))
  }
  if (op == "predict") {
    return(.opnclp_predict(payload$fit, payload$compute))
  }
  if (op == "compare_scaling") {
    la <- if (is.null(payload$label_a)) "A" else payload$label_a
    lb <- if (is.null(payload$label_b)) "B" else payload$label_b
    return(compare_scaling(payload$x_a, payload$y_a,
                           payload$x_b, payload$y_b, la, lb))
  }
  if (op == "infonce") {
    t <- if (is.null(payload$temperature)) 0.07 else payload$temperature
    return(infonce(payload$image_embeddings, payload$text_embeddings, t))
  }
  if (op == "cheatsheet") {
    return(.opnclp_cheatsheet())
  }
  stop("opnclp: unknown op")
}

#' .as_num_vec
#'
#' A step of the opnclp_native implementation. Called by \code{fit_power_law}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param p Optional; may be \code{NULL}. A list; the body checks with \code{is.list}.
#' @return A vector, from \code{as.numeric}.
#' @export
.as_num_vec <- function(p) {
  if (is.null(p)) return(numeric(0))
  if (is.list(p)) {
    out <- c()
    for (q in p) out <- c(out, as.numeric(q))
    return(out)
  }
  as.numeric(p)
}

#' .as_num_mat
#'
#' A step of the opnclp_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param m Optional; may be \code{NULL}. A matrix; the body checks with \code{is.matrix}.
#' @return The value of \code{do.call}.
#' @export
.as_num_mat <- function(m) {
  if (is.matrix(m)) return(m)
  if (is.null(m)) return(matrix(numeric(0), 0, 0))
  rows <- lapply(m, function(r) as.numeric(unlist(r)))
  do.call(rbind, rows)
}

#' total_compute
#'
#' A step of the opnclp_native implementation. Called by \code{morie_opnclp}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param samples_seen Coerced to numeric by the body, with \code{as.numeric}.
#' @param model_params Coerced to numeric by the body, with \code{as.numeric}.
#' @return A list with \code{compute}, \code{samples_seen}, \code{params}, \code{gmac_scale}.
#' @export
total_compute <- function(samples_seen, model_params) {
  s <- as.numeric(samples_seen)
  p <- as.numeric(model_params)
  if (s <= 0.0 || p <= 0.0) {
    stop("opnclp: both quantities must be positive")
  }
  list(compute = s * p, samples_seen = s, params = p,
       gmac_scale = s * p / 1e9)
}

#' fit_power_law
#'
#' A step of the opnclp_native implementation. Called by \code{compare_scaling}, \code{morie_opnclp}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param x Passed to \code{.as_num_vec}.
#' @param y Passed to \code{.as_num_vec}.
#' @return A list with \code{alpha}, \code{beta}, \code{slope}, \code{r_squared}, \code{range}, \code{n}.
#' @export
fit_power_law <- function(x, y) {
  X <- .as_num_vec(x)
  Y <- .as_num_vec(y)
  if (length(X) != length(Y)) {
    stop(sprintf("opnclp: %d x values but %d y values", length(X), length(Y)))
  }
  if (length(X) < 2L) {
    stop("opnclp: at least 2 points are needed")
  }
  if (any(X <= 0.0) || any(Y <= 0.0)) {
    stop("opnclp: a power law is fitted on the logs, so both axes must be strictly positive")
  }
  lx <- log(X)
  ly <- log(Y)
  n <- length(lx)
  mx <- sum(lx) / n
  my <- sum(ly) / n
  sxx <- sum((lx - mx) ^ 2)
  if (sxx <= 1e-12) {
    stop("opnclp: every x is the same, so no slope is identified")
  }
  sxy <- sum((lx - mx) * (ly - my))
  slope <- sxy / sxx
  inter <- my - slope * mx
  pred <- inter + slope * lx
  ss_res <- sum((ly - pred) ^ 2)
  ss_tot <- sum((ly - my) ^ 2)
  list(alpha = -slope, beta = exp(inter),
       slope = slope,
       r_squared = if (ss_tot > 1e-12) 1.0 - ss_res / ss_tot else 1.0,
       range = c(min(X), max(X)), n = n)
}

#' .opnclp_predict
#'
#' A step of the opnclp_native implementation. Called by \code{morie_opnclp}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param fit A list; the body reads \code{$alpha}, \code{$beta}, \code{$range} from it.
#' @param compute Coerced to numeric by the body, with \code{as.numeric}.
#' @return A list with \code{value}, \code{extrapolation_decades}, \code{interpolated}, \code{note}.
#' @export
.opnclp_predict <- function(fit, compute) {
  c <- as.numeric(compute)
  if (c <= 0.0) {
    stop("opnclp: compute must be positive")
  }
  rng <- fit$range
  lo <- rng[1]
  hi <- rng[2]
  decades <- if (c < lo) log10(lo / c) else if (c > hi) log10(c / hi) else 0.0
  list(value = fit$beta * c ^ (-fit$alpha),
       extrapolation_decades = decades,
       interpolated = decades == 0.0,
       note = paste("an extrapolation is a different claim from an",
                    "interpolation, so the distance is reported"))
}

#' compare_scaling
#'
#' A step of the opnclp_native implementation. Called by \code{morie_opnclp}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param x_a Passed to \code{fit_power_law}.
#' @param y_a Passed to \code{fit_power_law}.
#' @param x_b Passed to \code{fit_power_law}.
#' @param y_b Passed to \code{fit_power_law}.
#' @param label_a Defaults to \code{"A"}.
#' @param label_b Defaults to \code{"B"}.
#' @return The value of \code{out}, as built in the body.
#' @export
compare_scaling <- function(x_a, y_a, x_b, y_b, label_a = "A", label_b = "B") {
  fa <- fit_power_law(x_a, y_a)
  fb <- fit_power_law(x_b, y_b)
  d <- abs(fa$alpha - fb$alpha)
  out <- list(estimate = d, alpha_gap = d)
  out[[label_a]] <- fa
  out[[label_b]] <- fb
  out$same_law <- d < 0.01
  out$method <- "power-law comparison across training distributions; Cherti et al. (2023)"
  out$note <- paste("a single exponent implies a universality the paper",
                    "denies; this is where the distribution shows up")
  out
}

# Build a row x col numeric matrix from a list-of-list input.
#' Build a row x col numeric matrix from a list-of-list input
#'
#' A step of the opnclp_native implementation. Called by \code{infonce}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param m Optional; may be \code{NULL}. A vector; its length is taken.
#' @return The value of \code{do.call}.
#' @export
.coerce_mat <- function(m) {
  if (is.matrix(m)) return(m)
  if (is.null(m) || length(m) == 0L) return(matrix(numeric(0), 0, 0))
  rows <- lapply(m, function(r) {
    if (is.list(r)) as.numeric(unlist(r)) else as.numeric(r)
  })
  do.call(rbind, rows)
}

#' infonce
#'
#' A step of the opnclp_native implementation. Called by \code{morie_opnclp}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param image_embeddings Passed to \code{.coerce_mat}.
#' @param text_embeddings Passed to \code{.coerce_mat}.
#' @param temperature Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{0.07}.
#' @return A list with \code{loss}, \code{image_to_text}, \code{text_to_image}, \code{logits}, \code{note}.
#' @export
infonce <- function(image_embeddings, text_embeddings, temperature = 0.07) {
  I <- .coerce_mat(image_embeddings)
  Tt <- .coerce_mat(text_embeddings)
  n <- nrow(I)
  if (nrow(Tt) != n) {
    stop(sprintf("opnclp: %d images but %d texts", n, nrow(Tt)))
  }
  t <- as.numeric(temperature)
  if (t <= 0.0) {
    stop("opnclp: the temperature must be positive")
  }
  nrm <- function(v) {
    m <- sqrt(sum(v * v))
    if (m <= 1e-12) stop("opnclp: a zero embedding has no direction")
    v / m
  }
  Iu <- t(apply(I, 1, nrm))
  Tu <- t(apply(Tt, 1, nrm))
  S <- matrix(0, nrow = n, ncol = n)
  for (i in seq_len(n)) {
    for (j in seq_len(n)) {
      S[i, j] <- sum(Iu[i, ] * Tu[j, ]) / t
    }
  }
  ce <- function(rows) {
    tot <- 0.0
    for (i in seq_len(n)) {
      m <- max(rows[i, ])
      z <- sum(exp(rows[i, ] - m))
      tot <- tot + -(rows[i, i] - m - log(z))
    }
    tot / n
  }
  li <- ce(S)
  lt <- ce(t(S))
  list(loss = 0.5 * (li + lt), image_to_text = li,
       text_to_image = lt, logits = S,
       note = paste("symmetric, so neither modality is the anchor"))
}

#' .opnclp_cheatsheet
#'
#' A step of the opnclp_native implementation. Called by \code{morie_opnclp}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @return A character value.
#' @export
.opnclp_cheatsheet <- function() {
  paste("opnclp: CLIP-scale laws had been measured on PRIVATE data and",
        "models; re-run on public LAION with an open implementation,",
        "up to 2B pairs, and the scaling is a POWER LAW across",
        "zero-shot classification, retrieval, linear probing and",
        "fine-tuning. The key finding is that the exponent MOVES:",
        "OpenAI and OpenCLIP models scale differently despite",
        "identical architectures and similar recipes, so the training",
        "DISTRIBUTION is part of the law. Fit by least squares on the",
        "logs; report how many decades beyond the fitted range a",
        "prediction reaches.")
}
