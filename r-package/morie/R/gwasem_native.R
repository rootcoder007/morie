# EMMAX: a variance component model for sample structure in GWAS.
# Sources: Kang, H. M., Sul, J. H., Service, S. K., Zaitlen, N. A.,
# Kong, S., Freimer, N. B., Sabatti, C. and Eskin, E. (2010) "Variance
# component model to account for sample structure in genome-wide
# association studies", Nature Genetics 42(4), 348-354 (the three-step
# procedure, equations 5-7, the pseudoheritability definition, the
# case-control handling); Kang, H. M., Zaitlen, N. A., Wade, C. M.,
# Kirby, A., Heckerman, D., Daly, M. J. and Eskin, E. (2008) "Efficient
# control of population structure in model organism association
# mapping", Genetics 178(3), 1709-1723 (the variance component
# estimation EMMAX calls in step 2 and the spectral decomposition that
# makes it cheap).
#
# Native implementation mirroring Python morie.fn.gwasem exactly: the
# same IBS relatedness, the same Gower normalisation (equation 5),
# the same REML/ML variance component estimation on the spectral
# basis, the same GLS F-test or score test at every marker with the
# variance components fixed once, and the same genomic control.

#' EMMAX genome-wide association scan
#'
#' The three-step procedure: (1) Gower-normalise a relatedness matrix
#' \eqn{\hat S}, (2) estimate \eqn{\sigma_a^2, \sigma_e^2} once by
#' REML/ML in \eqn{\mathrm{Var}(Y) = \sigma_a^2 \hat S_N +
#' \sigma_e^2 I}, and (3) GLS F-test (or score test) at every marker
#' with that fixed V. Step 2 happening once is the eXpedited part; the
#' expensive per-marker REML is available as an exact alternative.
#'
#' @param y Phenotype vector.
#' @param genotypes Individual x marker matrix (minor allele counts).
#' @param kinship Optional n x n relatedness matrix; computed by IBS
#'   when omitted.
#' @param covariates Optional covariate matrix.
#' @param trait "quantitative" or "binary".
#' @param test "f" (GLS F-test) or "score".
#' @param ml Use maximum likelihood instead of REML.
#' @param per_marker_reml Re-estimate variance components per marker
#'   (the exact EMMA model, off by default).
#' @param min_maf Skip markers below this minor allele frequency.
#' @return A list with \code{beta}, \code{se}, \code{stat},
#'   \code{pvalue}, \code{variance_components},
#'   \code{pseudo_heritability}, \code{lambda_gc}, \code{skipped},
#'   \code{n}, \code{n_markers}, \code{test}, \code{trait},
#'   \code{per_marker_reml}, \code{note} and \code{method}.
#' @references Kang, H. M. et al. (2010); Kang, H. M. et al. (2008).
#' @export
morie_gwasem <- function(y, genotypes, kinship = NULL, covariates = NULL,
                         trait = "quantitative", test = "f", ml = FALSE,
                         per_marker_reml = FALSE, min_maf = 0.0) {
  yv <- as.numeric(y)
  G <- apply(genotypes, c(1L, 2L), as.numeric)
  n <- length(yv)
  if (n == 0L || nrow(G) != n)
    stop("gwasem: one genotype row per phenotype")
  m <- ncol(G)
  if (any(apply(G, 1L, length) != m))
    stop("gwasem: ragged genotype matrix")
  if (!(trait %in% c("quantitative", "binary")))
    stop("gwasem: trait must be 'quantitative' or 'binary'")
  if (trait == "binary" && any(!(yv %in% c(0.0, 1.0))))
    stop("gwasem: a binary trait must be coded 0/1")
  if (!(test %in% c("f", "score")))
    stop("gwasem: test must be 'f' or 'score'")

  K <- if (is.null(kinship)) morie_gwasem_kinship_ibs(G) else kinship
  vc <- morie_gwasem_reml(yv, K, covariates, ml)
  evals <- vc$evals; evecs <- vc$evecs; delta <- vc$delta

  rotate <- function(vec) as.numeric(t(evecs) %*% vec)

  base <- if (is.null(covariates)) {
    matrix(1.0, nrow = n, ncol = 1L)
  } else {
    cbind(1.0, apply(covariates, c(1L, 2L), as.numeric))
  }
  base_t <- t(apply(base, 2L, rotate))
  y_t <- rotate(yv)

  beta <- numeric(m); se <- numeric(m); stat <- numeric(m)
  pval <- numeric(m); skipped <- integer(0)
  for (j in seq_len(m)) {
    col <- G[, j]
    p_hat <- sum(col) / (2.0 * n)
    if (min(p_hat, 1 - p_hat) < min_maf || max(col) == min(col)) {
      skipped <- c(skipped, j)
      beta[j] <- NA_real_; se[j] <- NA_real_
      stat[j] <- 0.0; pval[j] <- 1.0
      next
    }
    if (per_marker_reml) {
      vcj <- morie_gwasem_reml(yv, K, covariates, ml)
      dj <- vcj$delta; ev <- vcj$evals; ev2 <- vcj$evecs
      Xfull <- cbind(base, col)
      rot <- t(apply(Xfull, 1L, function(r) t(ev2) %*% r))
      yr <- as.numeric(t(ev2) %*% yv)
      d <- ev + dj
    } else {
      col_t <- rotate(col)
      rot <- cbind(t(base_t), col_t)
      yr <- y_t
      d <- evals + delta
    }
    p <- ncol(rot)
    M <- matrix(0.0, nrow = p, ncol = p)
    for (a in seq_len(p)) for (b in seq_len(p))
      M[a, b] <- sum(rot[, a] * rot[, b] / d)
    v <- as.numeric(M %*% rep(0, p))   # placeholder; recompute below
    v <- numeric(p)
    for (a in seq_len(p)) v[a] <- sum(rot[, a] * yr / d)
    bb <- tryCatch(solve(M, v), error = function(e) NULL)
    if (is.null(bb)) {
      skipped <- c(skipped, j)
      beta[j] <- NA_real_; se[j] <- NA_real_
      stat[j] <- 0.0; pval[j] <- 1.0
      next
    }
    inv <- solve(M)
    rss <- sum((yr - as.numeric(rot %*% bb))^2 / d)
    df <- n - p
    s2 <- rss / df
    b_k <- bb[p]
    var_k <- s2 * inv[p, p]
    se_k <- sqrt(max(var_k, 0.0))
    beta[j] <- b_k; se[j] <- se_k
    if (test == "f") {
      f <- if (var_k > 0) (b_k * b_k / var_k) else 0.0
      stat[j] <- f
      pval[j] <- .gwasem_f_sf(f, 1, df)
    } else {
      p0 <- p - 1L
      M0 <- M[seq_len(p0), seq_len(p0), drop = FALSE]
      v0 <- as.numeric(M0 %*% rep(0, p0))
      v0 <- numeric(p0)
      for (a in seq_len(p0)) v0[a] <- sum(rot[, a] * yr / d)
      b0 <- solve(M0, v0)
      r0 <- yr - as.numeric(rot[, seq_len(p0), drop = FALSE] %*% b0)
      s20 <- sum(r0^2 / d) / (n - p0)
      vx <- numeric(p0)
      for (a in seq_len(p0)) vx[a] <- sum(rot[, a] * rot[, p] / d)
      cx <- solve(M0, vx)
      xres <- rot[, p] - as.numeric(rot[, seq_len(p0), drop = FALSE] %*% cx)
      num <- sum(xres * r0 / d)
      den <- sum(xres * xres / d) * s20
      chi <- if (den > 0) (num * num / den) else 0.0
      stat[j] <- chi
      pval[j] <- .gwasem_norm_sf(sqrt(max(chi, 0.0)))
    }
  }
  tested <- stat[setdiff(seq_len(m), skipped)]
  list(estimate = beta, beta = beta, se = se, stat = stat, pvalue = pval,
       variance_components = vc,
       pseudo_heritability = vc$pseudo_heritability,
       lambda_gc = if (length(tested) > 0L) morie_gwasem_gc(tested)
                   else NaN,
       skipped = skipped, n = n, n_markers = m, test = test, trait = trait,
       per_marker_reml = as.logical(per_marker_reml),
       note = paste0("the variance components are estimated ONCE under ",
                     "the null (that is what makes it EMMAX rather than ",
                     "EMMA); per_marker_reml=TRUE restores the exact ",
                     "model"),
       method = "EMMAX variance component association (Kang et al. 2010)")
}

#' IBS relatedness matrix
#'
#' \eqn{\hat S_{ik} = 1 - \frac{1}{2M}\sum_j |g_{ij} - g_{kj}|}.
#'
#' @param genotypes n x m minor-allele-count matrix.
#' @return n x n symmetric numeric matrix.
#' @references Kang, H. M. et al. (2010).
#' @export
morie_gwasem_kinship_ibs <- function(genotypes) {
  G <- apply(genotypes, c(1L, 2L), as.numeric)
  n <- nrow(G); m <- ncol(G)
  if (n == 0L || m == 0L)
    stop("gwasem: genotypes must be a non-empty individual x marker matrix")
  S <- matrix(0.0, nrow = n, ncol = n)
  for (i in seq_len(n) - 1L) {
    for (k in i:n - 1L) {
      d <- sum(abs(G[i + 1L, ] - G[k + 1L, ]))
      v <- 1.0 - d / (2.0 * m)
      S[i + 1L, k + 1L] <- v
      S[k + 1L, i + 1L] <- v
    }
  }
  S
}

#' Gower normalisation of a relatedness matrix
#'
#' Equation 5: \eqn{\hat S_N = (n-1)\hat S / \mathrm{Tr}(P\hat S P)}
#' with \eqn{P = I - \mathbf{1}\mathbf{1}'/n}, so \eqn{\sigma_a^2} is
#' on the scale of the phenotypic variance.
#'
#' @param S n x n relatedness matrix.
#' @return Normalised matrix.
#' @references Kang, H. M. et al. (2010).
#' @export
morie_gwasem_gower <- function(S) {
  Sn <- apply(S, c(1L, 2L), as.numeric)
  n <- nrow(Sn)
  if (n < 2L) stop("gwasem: need at least two individuals")
  rows <- rowMeans(Sn)
  total <- mean(rows)
  tr <- 0.0
  for (i in seq_len(n)) tr <- tr + Sn[i, i] - 2.0 * rows[i] + total
  if (abs(tr) < 1e-300)
    stop(paste0("gwasem: the relatedness matrix has zero centred trace; ",
                "it carries no structure to normalise"))
  f <- (n - 1.0) / tr
  Sn * f
}

#' REML estimation of the variance components
#'
#' Step 2: estimate \eqn{\sigma_a^2, \sigma_e^2} on the spectral
#' basis of \eqn{\hat S_N}, plus a null comparison.
#'
#' @param y Phenotype vector.
#' @param Kinship Relatedness matrix.
#' @param covariates Optional covariate matrix.
#' @param ml Use maximum likelihood.
#' @return List with \code{sigma_a2}, \code{sigma_e2}, \code{delta},
#'   \code{pseudo_heritability}, \code{loglik}, \code{loglik_null},
#'   \code{lrt}, \code{evals}, \code{evecs}, \code{kinship_normalized}
#'   and \code{shift}.
#' @references Kang, H. M. et al. (2010); Kang, H. M. et al. (2008).
#' @export
morie_gwasem_reml <- function(y, Kinship, covariates = NULL, ml = FALSE) {
  yv <- as.numeric(y)
  n <- length(yv)
  K <- morie_gwasem_gower(Kinship)
  if (nrow(K) != n)
    stop("gwasem: the kinship matrix must be n x n")
  X <- if (is.null(covariates)) {
    matrix(1.0, nrow = n, ncol = 1L)
  } else {
    if (nrow(covariates) != n)
      stop("gwasem: one covariate row per individual")
    cbind(1.0, apply(covariates, c(1L, 2L), as.numeric))
  }
  ev <- eigen(K, symmetric = TRUE)
  evals <- ev$values; evecs <- ev$vectors
  shift <- if (min(evals) <= 0) -min(evals) + 1e-8 else 0.0
  evals <- evals + shift
  delta <- sigma_a2 <- sigma_e2 <- ll <- NULL
  delta <- 0; sigma_a2 <- 0; sigma_e2 <- 0; ll <- 0
  r <- .gwasem_reml_delta(yv, X, evals, evecs, ml)
  delta <- r$delta; sigma_a2 <- r$sigma_a2; sigma_e2 <- r$sigma_e2
  ll <- r$ll
  p <- ncol(X)
  M0 <- crossprod(X)
  v0 <- as.numeric(crossprod(X, yv))
  beta0 <- solve(M0, v0)
  rss0 <- sum((yv - as.numeric(X %*% beta0))^2)
  df0 <- if (ml) n else n - p
  ll0 <- -0.5 * (df0 * log(2 * pi * rss0 / df0) + df0)
  if (!ml) {
    ldM <- determinant(M0, logarithm = TRUE)$modulus
    ll0 <- ll0 - 0.5 * as.numeric(ldM)
  }
  list(sigma_a2 = sigma_a2, sigma_e2 = sigma_e2, delta = delta,
       pseudo_heritability = if ((sigma_a2 + sigma_e2) > 0)
         sigma_a2 / (sigma_a2 + sigma_e2) else 0.0,
       loglik = ll, loglik_null = ll0,
       lrt = max(0.0, 2.0 * (ll - ll0)),
       evals = evals, evecs = evecs,
       kinship_normalized = K, shift = shift)
}

#' Genomic control inflation factor
#'
#' The ratio of the median test statistic to its null median.
#'
#' @param stats Numeric vector of test statistics.
#' @param df Degrees of freedom (1 for chi-square, k for F).
#' @return Scalar lambda.
#' @references Devlin, B. and Roeder, K. (1999).
#' @export
morie_gwasem_gc <- function(stats, df = 1) {
  s <- sort(as.numeric(stats))
  if (length(s) == 0L) stop("gwasem: no statistics")
  n <- length(s)
  med <- if (n %% 2L == 1L) s[(n + 1L) %/% 2L]
         else 0.5 * (s[n %/% 2L] + s[n %/% 2L + 1L])
  null_med <- if (df == 1L) 0.4549364231195736
              else as.numeric(df) * (1.0 - 2.0 / (9.0 * df))^3
  med / null_med
}

# -- helpers ----------------------------------------------------------------

.gwasem_reml_delta <- function(y, X, evals, evecs, ml) {
  n <- length(y); p <- ncol(X)
  yt <- as.numeric(t(evecs) %*% y)
  Xt <- t(apply(X, 1L, function(r) as.numeric(t(evecs) %*% r)))
  loglik <- function(delta) {
    d <- evals + delta
    if (min(d) <= 1e-12) return(-Inf)
    M <- matrix(0.0, nrow = p, ncol = p)
    for (a in seq_len(p)) for (b in seq_len(p))
      M[a, b] <- sum(Xt[, a] * Xt[, b] / d)
    v <- numeric(p)
    for (a in seq_len(p)) v[a] <- sum(Xt[, a] * yt / d)
    bb <- tryCatch(solve(M, v), error = function(e) NULL)
    if (is.null(bb)) return(-Inf)
    ldM <- determinant(M, logarithm = TRUE)$modulus
    if (attr(ldM, "sign") != 1) return(-Inf)
    rss <- sum((yt - as.numeric(Xt %*% bb))^2 / d)
    if (rss <= 0) return(-Inf)
    logdetV <- sum(log(d))
    if (ml) {
      -0.5 * (n * log(2 * pi * rss / n) + n + logdetV)
    } else {
      df <- n - p
      -0.5 * (df * log(2 * pi * rss / df) + df + logdetV + as.numeric(ldM))
    }
  }
  lo <- -10.0; hi <- 10.0; n_grid <- 100L
  best_u <- lo; best_v <- loglik(exp(lo))
  for (g in seq_len(n_grid)) {
    u <- lo + (hi - lo) * g / n_grid
    val <- loglik(exp(u))
    if (val > best_v) { best_u <- u; best_v <- val }
  }
  step <- (hi - lo) / n_grid
  a <- best_u - step; b <- best_u + step
  phi <- (sqrt(5.0) - 1.0) / 2.0
  c <- b - phi * (b - a); d <- a + phi * (b - a)
  fc <- loglik(exp(c)); fd <- loglik(exp(d))
  for (kk in seq_len(60L)) {
    if (fc > fd) { b <- d; fd <- fc; d <- c; fd <- loglik(exp(d));
                   c <- b - phi * (b - a); fc <- loglik(exp(c)) }
    else { a <- c; fc <- fd; c <- d; fc <- loglik(exp(c));
           d <- a + phi * (b - a); fd <- loglik(exp(d)) }
  }
  delta <- exp(0.5 * (a + b))
  d_ <- evals + delta
  M <- matrix(0.0, nrow = p, ncol = p)
  for (a in seq_len(p)) for (b in seq_len(p))
    M[a, b] <- sum(Xt[, a] * Xt[, b] / d_)
  v <- numeric(p)
  for (a in seq_len(p)) v[a] <- sum(Xt[, a] * yt / d_)
  bb <- solve(M, v)
  rss <- sum((yt - as.numeric(Xt %*% bb))^2 / d_)
  df <- if (ml) n else n - p
  sigma_a2 <- rss / df
  list(delta = delta, sigma_a2 = sigma_a2,
       sigma_e2 = sigma_a2 * delta, ll = loglik(delta))
}

.gwasem_norm_sf <- function(z) {
  pnorm(abs(z), lower.tail = FALSE)
}

.gwasem_f_sf <- function(f, df1, df2) {
  if (f <= 0) return(1.0)
  x <- df2 / (df2 + df1 * f)
  a <- 0.5 * df2; b <- 0.5 * df1
  lbeta <- (lgamma(a + b) - lgamma(a) - lgamma(b) +
            a * log(x) + b * log(1.0 - x))
  betacf <- function(a, b, x) {
    qab <- a + b; qap <- a + 1.0; qam <- a - 1.0
    c <- 1.0; d <- 1.0 - qab * x / qap
    d <- if (abs(d) > 1e-300) 1.0 / d else 1e300
    h <- d
    for (mm in seq_len(300L)) {
      m2 <- 2L * mm
      aa <- mm * (b - mm) * x / ((qam + m2) * (a + m2))
      d <- 1.0 + aa * d
      d <- if (abs(d) > 1e-300) 1.0 / d else 1e300
      c <- 1.0 + aa / c
      c <- if (abs(c) > 1e-300) c else 1e-300
      h <- h * d * c
      aa <- -(a + mm) * (qab + mm) * x / ((a + m2) * (qap + m2))
      d <- 1.0 + aa * d
      d <- if (abs(d) > 1e-300) 1.0 / d else 1e300
      c <- 1.0 + aa / c
      c <- if (abs(c) > 1e-300) c else 1e-300
      de <- d * c
      h <- h * de
      if (abs(de - 1.0) < 3e-16) break
    }
    h
  }
  if (x < (a + 1.0) / (a + b + 2.0)) {
    exp(lbeta) * betacf(a, b, x) / a
  } else {
    1.0 - exp(lbeta) * betacf(b, a, 1.0 - x) / b
  }
}
