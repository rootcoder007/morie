# mqtmpl -- genome scans for QTL (R/qtl style)
# References:
#   Broman et al. (2003) "R/qtl" Bioinformatics 19(7), 889-890
#   Lander & Botstein (1989) Genetics 121(1), 185-199
#   Churchill & Doerge (1994) Genetics 138(3), 963-971
#   Baum et al. (1970) Ann. Math. Statist. 41(1), 164-171
#   Sen & Churchill (2001) Genetics 159(1), 371-387
# Base R only.

mqtmpl_LOG10E <- log10(exp(1))

#' mqtmpl_haldane
#'
#' A step of the mqtmpl_native implementation. Called by \code{mqtmpl_hmm_genotype_probabilities}, \code{mqtmpl_sample_genotypes}, \code{mqtmpl_scan_cim}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param d Numeric; combined arithmetically in the body.
#' @return A numeric value.
#' @export
mqtmpl_haldane <- function(d) {
  d <- as.numeric(d)
  0.5 * (1 - exp(-2 * d / 100))
}

#' Qk left, qk+1 right
#'
#' A step of the mqtmpl_native implementation. Called by \code{mqtmpl_cim_one}, \code{mqtmpl_sample_genotypes}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param s_left See Usage.
#' @param s_right See Usage.
#' @param r_left Numeric; combined arithmetically in the body.
#' @param r_right Numeric; combined arithmetically in the body.
#' @return A vector, from \code{c}.
#' @export
mqtmpl_genotype_probabilities <- function(s_left, s_right, r_left, r_right) {
  # qk left, qk+1 right
  p1 <- (1 - r_left) * (1 - r_right)
  p2 <- r_left * r_right
  n_tot <- p1 + p2
  if (n_tot <= 0) return(c(0, 0))
  c(p2 / n_tot, p1 / n_tot)
}

#' mqtmpl_single_marker
#'
#' A step of the mqtmpl_native implementation. Called by \code{mqtmpl_scanone}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param y A vector; its length is taken.
#' @param g Numeric; passed to \code{mean}.
#' @return A list with \code{lod}, \code{rss}, \code{rss0}.
#' @export
mqtmpl_single_marker <- function(y, g) {
  n <- length(y)
  my <- mean(y); mg <- mean(g)
  sgg <- sum((g - mg)^2)
  if (sgg <= 0) {
    rss <- sum((y - my)^2)
  } else {
    b <- sum((g - mg) * (y - my)) / sgg
    a <- my - b * mg
    rss <- sum((y - (a + b * g))^2)
  }
  rss <- max(rss, 1e-300)
  rss0 <- sum((y - my)^2)
  rss0 <- max(rss0, 1e-300)
  lod <- 0.5 * (n * log(rss0) - n * log(rss) - log(n)) * mqtmpl_LOG10E
  list(lod = lod, rss = rss, rss0 = rss0)
}

#' mqtmpl_cim_one
#'
#' A step of the mqtmpl_native implementation. Called by \code{mqtmpl_scan_cim}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param y A vector; its length is taken.
#' @param left See Usage.
#' @param right See Usage.
#' @param r_left See Usage.
#' @param r_right See Usage.
#' @param cofactors See Usage.
#' @return A list with \code{lod}, \code{rss}, \code{coef}.
#' @export
mqtmpl_cim_one <- function(y, left, right, r_left, r_right, cofactors) {
  n <- length(y)
  pr <- mqtmpl_genotype_probabilities(left, right, r_left, r_right)
  # Q matrix
  Q <- cbind(rep(1, n), pr[1], pr[2])
  for (co in cofactors) Q <- cbind(Q, co)
  qa <- qr(Q)
  coef <- as.numeric(qr.coef(qa, y))
  fitted <- as.numeric(Q %*% coef)
  rss <- sum((y - fitted)^2)
  rss0 <- sum((y - mean(y))^2)
  rss <- max(rss, 1e-300); rss0 <- max(rss0, 1e-300)
  v <- ncol(Q)
  lod <- 0.5 * (n * log(rss0) - n * log(rss) - (v - 1) * log(n)) * mqtmpl_LOG10E
  list(lod = lod, rss = rss, coef = coef)
}

#' mqtmpl_scan_cim
#'
#' A step of the mqtmpl_native implementation. Called by \code{mqtmpl_scanone}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param y See Usage.
#' @param markers A vector; its length is taken and its elements indexed.
#' @param positions A vector; indexed elementwise.
#' @param cofactors A vector; its length is taken. Defaults to \code{list()}.
#' @param window Defaults to \code{0}.
#' @param step Numeric; combined arithmetically in the body. Defaults to \code{0.02}.
#' @return A list with \code{estimate}, \code{peak_lod}, \code{peak_position}, \code{position}, \code{lod}, \code{fit}.
#' @export
mqtmpl_scan_cim <- function(y, markers, positions, cofactors = list(),
                            window = 0, step = 0.02) {
  m <- length(markers)
  out_pos <- c(); out_lod <- c(); fits <- list()
  if (length(cofactors) == 0L) {
    for (j in seq_len(m - 1L)) {
      span <- positions[j + 1L] - positions[j]
      d <- 0
      while (d <= span + 1e-12) {
        r_left <- mqtmpl_haldane(min(d, span))
        r_right <- mqtmpl_haldane(max(span - d, 0))
        f <- mqtmpl_cim_one(y, markers[[j]], markers[[j + 1L]],
                            r_left, r_right, cofactors)
        out_pos <- c(out_pos, positions[j] + d)
        out_lod <- c(out_lod, f$lod)
        fits[[length(fits) + 1L]] <- f
        d <- d + step
      }
    }
  } else {
    for (j in seq_len(m - 1L)) {
      span <- positions[j + 1L] - positions[j]
      d <- 0
      while (d <= span + 1e-12) {
        r_left <- mqtmpl_haldane(min(d, span))
        r_right <- mqtmpl_haldane(max(span - d, 0))
        f <- mqtmpl_cim_one(y, markers[[j]], markers[[j + 1L]],
                            r_left, r_right, cofactors)
        out_pos <- c(out_pos, positions[j] + d)
        out_lod <- c(out_lod, f$lod)
        fits[[length(fits) + 1L]] <- f
        d <- d + step
      }
    }
  }
  k <- which.max(out_lod)
  list(estimate = out_lod[k], peak_lod = out_lod[k],
       peak_position = out_pos[k], position = out_pos, lod = out_lod,
       fit = fits[[k]])
}

mqtmpl_METHODS <- c("em", "mr", "hk", "imp")
mqtmpl_AVAILABLE <- c("em", "mr", "imp")
mqtmpl_UNSOURCED <- list(
  hk = "Haley-Knott regression is named but not defined in Broman et al. (2003); the primary source, Haley, C. S. & Knott, S. A. (1992) 'A simple regression method for mapping quantitative trait loci in line crosses using flanking markers', Heredity 69(4), 315-324, doi:10.1038/hdy.1992.131, is not in the corpus"
)

#' mqtmpl_method_status
#'
#' A step of the mqtmpl_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param method Defaults to \code{NULL}.
#' @return A list with \code{method}, \code{available}, \code{reason}.
#' @export
mqtmpl_method_status <- function(method = NULL) {
  if (is.null(method)) {
    return(list(methods = mqtmpl_METHODS, available = mqtmpl_AVAILABLE,
                unavailable = mqtmpl_UNSOURCED))
  }
  if (!(method %in% mqtmpl_METHODS)) {
    stop(sprintf("mqtmpl: method must be one of %s, got %s",
                 paste(mqtmpl_METHODS, collapse = ", "), method))
  }
  list(method = method, available = method %in% mqtmpl_AVAILABLE,
       reason = if (method %in% names(mqtmpl_UNSOURCED)) mqtmpl_UNSOURCED[[method]] else "")
}

#' mqtmpl_check_method
#'
#' A step of the mqtmpl_native implementation. Called by \code{mqtmpl_scanone}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param method See Usage.
#' @return One of two values, depending on the branch taken.
#' @export
mqtmpl_check_method <- function(method) {
  if (!(method %in% mqtmpl_METHODS)) {
    stop(sprintf("mqtmpl: method must be one of %s, got %s",
                 paste(mqtmpl_METHODS, collapse = ", "), method))
  }
  if (!(method %in% mqtmpl_AVAILABLE)) {
    stop(sprintf("mqtmpl: the '%s' scan method is not implemented -- %s",
                 method, mqtmpl_UNSOURCED[[method]]))
  }
}

#' mqtmpl_hmm_genotype_probabilities
#'
#' A step of the mqtmpl_native implementation. Called by \code{mqtmpl_sample_genotypes}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param genotypes See Usage.
#' @param positions A vector; its length is taken and its elements indexed.
#' @param error_rate Defaults to \code{0}.
#' @return The value of \code{out}, as built in the body.
#' @export
mqtmpl_hmm_genotype_probabilities <- function(genotypes, positions, error_rate = 0) {
  e <- as.numeric(error_rate)
  if (!(e >= 0 && e < 0.5)) {
    stop(sprintf("mqtmpl: the genotyping error rate must lie in [0, 0.5), got %g", e))
  }
  m <- length(positions)
  if (any(sapply(genotypes, length) != m)) {
    stop("mqtmpl: every individual needs one call per marker")
  }
  trans <- list()
  for (j in seq_len(m - 1L)) {
    d <- positions[j + 1L] - positions[j]
    if (d <= 0) stop("mqtmpl: marker positions must increase")
    trans[[j]] <- mqtmpl_haldane(d)
  }
  out <- list()
  for (row in genotypes) {
    emit <- function(j, state) {
      v <- row[j]
      if (is.null(v) || is.na(v)) return(1)
      if (as.integer(v) == state) 1 - e else e
    }
    f <- matrix(0, nrow = m, ncol = 2)
    f[1L, 1L] <- 0.5 * emit(1L, 0L)
    f[1L, 2L] <- 0.5 * emit(1L, 1L)
    for (j in seq.int(2L, m)) {
      r <- trans[[j - 1L]]
      for (s in 0:1) {
        f[j, s + 1L] <- emit(j, s) * (
          f[j - 1L, s + 1L] * (1 - r) + f[j - 1L, 2L - s] * r)
      }
      tot <- sum(f[j, ])
      if (tot <= 0) stop("mqtmpl: an individual's marker data have probability zero; raise the error rate")
      f[j, ] <- f[j, ] / tot
    }
    b <- matrix(0, nrow = m, ncol = 2)
    b[m, ] <- c(1, 1)
    for (j in seq.int(m - 1L, 1L)) {
      r <- trans[[j]]
      for (s in 0:1) {
        b[j, s + 1L] <- (b[j + 1L, s + 1L] * (1 - r) * emit(j + 1L, s) +
                         b[j + 1L, 2L - s] * r * emit(j + 1L, 1L - s))
      }
      tot <- sum(b[j, ])
      if (tot > 0) b[j, ] <- b[j, ] / tot
    }
    post <- matrix(0, nrow = m, ncol = 2)
    for (j in seq_len(m)) {
      p0 <- f[j, 1L] * b[j, 1L]
      p1 <- f[j, 2L] * b[j, 2L]
      s <- p0 + p1
      post[j, ] <- c(p0 / s, p1 / s)
    }
    out[[length(out) + 1L]] <- post
  }
  out
}

#' mqtmpl_sample_genotypes
#'
#' A step of the mqtmpl_native implementation. Called by \code{mqtmpl_scan_imp}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param genotypes A vector; its length is taken.
#' @param positions A vector; its length is taken and its elements indexed.
#' @param grid A vector; its length is taken and its elements indexed.
#' @param n_imp Defaults to \code{16}.
#' @param error_rate Defaults to \code{0}.
#' @param seed Defaults to \code{0}.
#' @return The value of \code{out}, as built in the body.
#' @export
mqtmpl_sample_genotypes <- function(genotypes, positions, grid, n_imp = 16,
                                    error_rate = 0, seed = 0) {
  m <- length(positions)
  n <- length(genotypes)
  set.seed(seed)
  post <- mqtmpl_hmm_genotype_probabilities(genotypes, positions, error_rate)
  out <- list()
  for (imp in seq_len(as.integer(n_imp))) {
    draw <- list()
    for (i in seq_len(n)) {
      states <- integer(m)
      states[m] <- if (runif(1) < post[[i]][m, 2L]) 1L else 0L
      for (j in seq.int(m - 1L, 1L)) {
        r <- mqtmpl_haldane(positions[j + 1L] - positions[j])
        w <- c(post[[i]][j, 1L] * (1 - r + r * (states[j + 1L] == 0L)),
               post[[i]][j, 2L] * (1 - r + r * (states[j + 1L] == 1L)))
        tot <- sum(w)
        states[j] <- if (runif(1) < w[2L] / tot) 1L else 0L
      }
      row <- numeric(length(grid))
      for (gi in seq_along(grid)) {
        g <- grid[gi]
        js <- which(positions <= g + 1e-12)
        j <- max(js)
        if (j == m) { row[gi] <- states[j]; next }
        d1 <- max(g - positions[j], 0)
        d2 <- max(positions[j + 1L] - g, 0)
        pr <- mqtmpl_genotype_probabilities(states[j], states[j + 1L],
                                            mqtmpl_haldane(d1),
                                            mqtmpl_haldane(d2))
        row[gi] <- if (runif(1) < pr[2L]) 1 else 0
      }
      draw[[i]] <- row
    }
    out[[length(out) + 1L]] <- draw
  }
  out
}

#' mqtmpl_imputation_weights
#'
#' A step of the mqtmpl_native implementation. Called by \code{mqtmpl_scan_imp}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param y A vector; its length is taken.
#' @param genotype_column A vector; its length is taken.
#' @param model_dimension Numeric; combined arithmetically in the body. Defaults to \code{2}.
#' @return A numeric value.
#' @export
mqtmpl_imputation_weights <- function(y, genotype_column, model_dimension = 2) {
  n <- length(y)
  if (n != length(genotype_column)) stop("mqtmpl: one genotype per phenotype")
  g <- as.numeric(genotype_column)
  my <- mean(y); mg <- mean(g)
  sgg <- sum((g - mg)^2)
  if (sgg <= 0) {
    rss <- sum((y - my)^2)
  } else {
    b <- sum((g - mg) * (y - my)) / sgg
    a <- my - b * mg
    rss <- sum((y - (a + b * g))^2)
  }
  rss <- max(rss, 1e-300)
  -0.5 * model_dimension * log(n) - 0.5 * n * log(rss)
}

#' mqtmpl_scan_imp
#'
#' A step of the mqtmpl_native implementation. Called by \code{mqtmpl_scanone}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param y A vector; its length is taken.
#' @param markers A vector; its length is taken and its elements indexed.
#' @param positions A vector; its length is taken and its elements indexed.
#' @param step Numeric; combined arithmetically in the body.
#' @param n_imp See Usage.
#' @param error_rate See Usage.
#' @param seed See Usage.
#' @return A list with \code{estimate}, \code{peak_lod}, \code{peak_position}, \code{position}, \code{lod}, \code{method_used}, \code{n_imputations}, \code{note}, \code{method}.
#' @export
mqtmpl_scan_imp <- function(y, markers, positions, step, n_imp,
                            error_rate, seed) {
  n <- length(y)
  grid <- c()
  g <- positions[1L]
  end <- positions[length(positions)]
  while (g <= end + 1e-12) { grid <- c(grid, g); g <- g + step }
  geno <- lapply(seq_len(n), function(i) {
    sapply(seq_along(markers), function(j) markers[[j]][i])
  })
  draws <- mqtmpl_sample_genotypes(geno, positions, grid, n_imp, error_rate, seed)
  null <- mqtmpl_imputation_weights(y, rep(0, n), model_dimension = 1)
  lods <- numeric(length(grid))
  for (gi in seq_along(grid)) {
    ws <- sapply(draws, function(d) {
      mqtmpl_imputation_weights(y, sapply(seq_len(n), function(i) d[[i]][gi]))
    })
    top <- max(ws)
    avg <- top + log(sum(exp(ws - top)) / length(ws))
    lods[gi] <- (avg - null) * mqtmpl_LOG10E
  }
  k <- which.max(lods)
  list(estimate = lods[k], peak_lod = lods[k], peak_position = grid[k],
       position = grid, lod = lods, method_used = "imp",
       n_imputations = as.integer(n_imp),
       note = "weights are n^(-v/2) RSS^(-n/2) on the log scale; the draws depend on the markers only, so a new model reuses them and only the weights change",
       method = "multiple-imputation scan; Sen & Churchill (2001) eqs (3)-(4)")
}

#' mqtmpl_kw_n_imp
#'
#' A step of the mqtmpl_native implementation. Called by \code{mqtmpl_scanone}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param covariates A vector; its length is taken.
#' @return A numeric value.
#' @export
mqtmpl_kw_n_imp <- function(covariates) {
  if (length(covariates) > 0L) {
    stop("mqtmpl: covariates are not implemented for the imputation scan")
  }
  64L
}

#' mqtmpl_scanone
#'
#' A step of the mqtmpl_native implementation. Called by \code{mqtmpl_permutation_threshold}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param y A vector; its length is taken and its elements indexed.
#' @param markers A vector; its length is taken and its elements indexed.
#' @param positions A vector; indexed elementwise.
#' @param method One of \code{"imp"}, \code{"mr"}. Defaults to \code{"em"}.
#' @param step Defaults to \code{0.02}.
#' @param covariates A vector; its length is taken. Defaults to \code{list()}.
#' @param error_rate Defaults to \code{0}.
#' @return A list with \code{estimate}, \code{peak_lod}, \code{peak_position}, \code{position}, \code{lod}, \code{method_used}, \code{n_covariates}, \code{error_rate}, \code{method}.
#' @export
mqtmpl_scanone <- function(y, markers, positions, method = "em", step = 0.02,
                           covariates = list(), error_rate = 0) {
  mqtmpl_check_method(method)
  n <- length(y)
  if (any(sapply(markers, length) != n)) {
    stop(sprintf("mqtmpl: every marker must be typed on all %d individuals", n))
  }
  if (method == "imp") {
    return(mqtmpl_scan_imp(y, markers, positions, step,
                           mqtmpl_kw_n_imp(covariates), error_rate, 0))
  }
  if (method == "mr") {
    out_pos <- c(); out_lod <- c()
    for (j in seq_along(markers)) {
      typed <- which(!sapply(markers[[j]], is.null))
      if (length(typed) < 3L) next
      sm <- mqtmpl_single_marker(y[typed], unlist(markers[[j]])[typed])
      out_pos <- c(out_pos, positions[j])
      out_lod <- c(out_lod, sm$lod)
    }
    if (length(out_lod) == 0L) {
      stop("mqtmpl: no marker has enough typed individuals")
    }
    k <- which.max(out_lod)
    return(list(estimate = out_lod[k], peak_lod = out_lod[k],
                peak_position = out_pos[k], position = out_pos,
                lod = out_lod, method_used = "mr",
                note = "marker regression reports LOD at markers only, and drops individuals not typed there",
                method = "marker regression scan; Broman et al. (2003)"))
  }
  # em
  res <- mqtmpl_scan_cim(y, markers, positions, covariates, 0, step)
  list(estimate = res$peak_lod, peak_lod = res$peak_lod,
       peak_position = res$peak_position, position = res$position,
       lod = res$lod, method_used = "em", n_covariates = length(covariates),
       error_rate = as.numeric(error_rate),
       method = "EM genome scan; Lander & Botstein (1989) via Broman et al. (2003)")
}

#' mqtmpl_permutation_threshold
#'
#' A step of the mqtmpl_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param y See Usage.
#' @param markers See Usage.
#' @param positions See Usage.
#' @param n_perm Defaults to \code{100}.
#' @param alpha Defaults to \code{0.05}.
#' @param method Defaults to \code{"em"}.
#' @param step Defaults to \code{0.05}.
#' @param seed Defaults to \code{0}.
#' @param ... Passed through.
#' @return A list with \code{estimate}, \code{threshold}, \code{alpha}, \code{n_perm}, \code{null_maxima}, \code{median_null}, \code{method}.
#' @export
mqtmpl_permutation_threshold <- function(y, markers, positions, n_perm = 100,
                                         alpha = 0.05, method = "em",
                                         step = 0.05, seed = 0, ...) {
  a <- as.numeric(alpha)
  if (!(a > 0 && a < 1)) stop("mqtmpl: alpha must lie in (0, 1)")
  set.seed(seed)
  maxima <- c()
  ys <- as.numeric(y)
  for (k in seq_len(as.integer(n_perm))) {
    perm <- sample(ys, length(ys))
    maxima <- c(maxima, mqtmpl_scanone(perm, markers, positions, method, step)$peak_lod)
  }
  maxima <- sort(maxima)
  idx <- min(length(maxima) - 1L,
             max(0L, as.integer(ceiling((1 - a) * length(maxima))) - 1L))
  list(estimate = maxima[idx + 1L], threshold = maxima[idx + 1L],
       alpha = a, n_perm = as.integer(n_perm), null_maxima = maxima,
       median_null = maxima[length(maxima) %/% 2L + 1L],
       method = "permutation threshold; Churchill & Doerge (1994) via Broman et al. (2003)")
}

#' mqtmpl_lod_support_interval
#'
#' A step of the mqtmpl_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param scan_result A list; the body reads \code{$lod}, \code{$position} from it.
#' @param drop Defaults to \code{1.5}.
#' @return A list with \code{peak}, \code{lower}, \code{upper}, \code{drop}, \code{peak_lod}.
#' @export
mqtmpl_lod_support_interval <- function(scan_result, drop = 1.5) {
  lod <- scan_result$lod
  pos <- scan_result$position
  k <- which.max(lod)
  cut <- lod[k] - as.numeric(drop)
  lo <- k
  while (lo > 1L && lod[lo - 1L] >= cut) lo <- lo - 1L
  hi <- k
  while (hi < length(lod) && lod[hi + 1L] >= cut) hi <- hi + 1L
  list(peak = pos[k], lower = pos[lo], upper = pos[hi],
       drop = as.numeric(drop), peak_lod = lod[k])
}

#' mqtmpl_cheatsheet
#'
#' A step of the mqtmpl_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @return A character value.
#' @export
mqtmpl_cheatsheet <- function() {
  paste("mqtmpl: the scanning layer. Genotypes come from a forward-backward HMM that tolerates missing calls and a genotyping error rate, and collapses to the flanking-marker formula when both are absent. Scans by EM or marker regression; Haley-Knott and multiple imputation are named and REFUSED, with citations. Genome-wide significance is a permutation threshold, because the maximum over correlated positions is not chi-squared anything.")
}

# house entry point: the package exports one morie_<module>
morie_mqtmpl <- mqtmpl_haldane
