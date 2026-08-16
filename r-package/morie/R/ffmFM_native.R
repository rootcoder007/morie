# morie.fn -- function file (rootcoder007/morie)
# Field-aware factorization machines.
#
# References
# Juan, Y., Zhuang, Y., Chin, W.-S. & Lin, C.-J. (2016) "Field-aware
# Factorization Machines for CTR Prediction", Proceedings of the Tenth
# ACM Conference on Recommender Systems (RecSys '16), 43-50,
# doi:10.1145/2959100.2959134. Sec. 2 (FM and its limitation), Sec. 3
# (the FFM model equation with the crossed field indices, the nfk
# parameter count, the logistic loss with y in {-1,1}, and the AdaGrad
# updates of eqs. (8)-(9)), and Sec. 3.3 (overfitting and early
# stopping).
# Rendle, S. (2010) "Factorization Machines", ICDM 2010, 995-1000,
# doi:10.1109/ICDM.2010.127. The model FFM specialises.

.ffmFM_EPS <- 1e-12

.n_parameters <- function(n_features, n_fields, k_dim, model = "ffm") {
  n <- as.integer(n_features); f <- as.integer(n_fields)
  kk <- as.integer(k_dim)
  if (!(model %in% c("ffm", "fm"))) {
    stop(sprintf("ffmFM: model must be ffm or fm, got %s", sQuote(model)))
  }
  if (model == "ffm") n * f * kk else n * kk
}

.ffmFM_phi <- function(x, fields, W) {
  nz <- list()
  for (kv in x) {
    j <- as.integer(kv[[1L]]); v <- as.numeric(kv[[2L]])
    if (v != 0) nz[[length(nz) + 1L]] <- list(j = j, v = v)
  }
  tot <- 0
  na <- length(nz)
  for (a in seq_len(na)) {
    for (b in (a + 1L):na) {
      j1 <- nz[[a]]$j; v1 <- nz[[a]]$v
      j2 <- nz[[b]]$j; v2 <- nz[[b]]$v
      f1 <- as.integer(fields[j1 + 1L]); f2 <- as.integer(fields[j2 + 1L])
      w1 <- W[[j1 + 1L]][[f2 + 1L]]; w2 <- W[[j2 + 1L]][[f1 + 1L]]
      kk <- length(w1)
      s <- 0
      for (d in seq_len(kk)) s <- s + w1[d] * w2[d]
      tot <- tot + s * v1 * v2
    }
  }
  tot
}

.logistic_loss <- function(y, phi_val) {
  yv <- as.numeric(y)
  if (!(yv == -1 || yv == 1)) {
    stop(sprintf("ffmFM: the label must be -1 or 1, got %s", format(y)))
  }
  z <- -yv * as.numeric(phi_val)
  if (z < 700) log(1 + exp(z)) else z
}

.fit_ffm <- function(rows, labels, fields, n_features, n_fields,
                     k_dim = 4L, eta = 0.1, lam = 2e-5, epochs = 10L,
                     seed = 0L) {
  n <- as.integer(n_features); F_ <- as.integer(n_fields)
  kk <- as.integer(k_dim)
  if (n < 1L || F_ < 1L || kk < 1L) {
    stop("ffmFM: n_features, n_fields and k must all be at least 1")
  }
  if (length(rows) != length(labels)) {
    stop(sprintf("ffmFM: %d rows but %d labels", length(rows), length(labels)))
  }
  rng <- .ghc_rng(as.integer(seed))
  scale <- 1 / sqrt(kk)
  W <- vector("list", n)
  G <- vector("list", n)
  raw <- .ghc_unif(rng, n * F_ * kk) * scale
  for (j in seq_len(n)) {
    Wj <- vector("list", F_)
    Gj <- vector("list", F_)
    for (f in seq_len(F_)) {
      lo <- ((j - 1L) * F_ + (f - 1L)) * kk
      Wj[[f]] <- raw[(lo + 1L):(lo + kk)]
      Gj[[f]] <- rep(1, kk)
    }
    W[[j]] <- Wj; G[[j]] <- Gj
  }
  hist <- numeric(0)
  for (ep in seq_len(as.integer(epochs))) {
    tot <- 0
    for (r in seq_along(rows)) {
      y <- as.numeric(labels[[r]])
      p <- .ffmFM_phi(rows[[r]], fields, W)
      tot <- tot + .logistic_loss(y, p)
      yp <- y * p
      g0 <- -y / (1 + exp(min(700, yp)))
      nz <- list()
      for (kv in rows[[r]]) {
        j <- as.integer(kv[[1L]]); v <- as.numeric(kv[[2L]])
        if (v != 0) nz[[length(nz) + 1L]] <- list(j = j, v = v)
      }
      na <- length(nz)
      for (a in seq_len(na)) {
        for (b in (a + 1L):na) {
          j1 <- nz[[a]]$j + 1L; v1 <- nz[[a]]$v
          j2 <- nz[[b]]$j + 1L; v2 <- nz[[b]]$v
          f1 <- as.integer(fields[j1]) + 1L
          f2 <- as.integer(fields[j2]) + 1L
          for (d in seq_len(kk)) {
            g1 <- lam * W[[j1]][[f2]][d] + g0 * W[[j2]][[f1]][d] * v1 * v2
            g2 <- lam * W[[j2]][[f1]][d] + g0 * W[[j1]][[f2]][d] * v1 * v2
            G[[j1]][[f2]][d] <- G[[j1]][[f2]][d] + g1 * g1
            G[[j2]][[f1]][d] <- G[[j2]][[f1]][d] + g2 * g2
            W[[j1]][[f2]][d] <- W[[j1]][[f2]][d] - eta / sqrt(G[[j1]][[f2]][d]) * g1
            W[[j2]][[f1]][d] <- W[[j2]][[f1]][d] - eta / sqrt(G[[j2]][[f1]][d]) * g2
          }
        }
      }
    }
    hist <- c(hist, tot / length(rows))
  }
  list(
    estimate = W, W = W, loss_history = hist, final_loss = hist[length(hist)],
    k = kk, n_parameters = .n_parameters(n, F_, kk),
    n_parameters_fm = .n_parameters(n, F_, kk, "fm"),
    method = "FFM with AdaGrad; Juan, Zhuang, Chin & Lin (2016) eqs. (8)-(9)",
    caveat = "FFM overfits readily -- the paper stops early on a validation set"
  )
}

#' morie_ffmFM
#'
#' Part of the ffmFM_native implementation; see the file header for the
#' source it follows.
#'
#' @param rows See Usage.
#' @param labels See Usage.
#' @param fields See Usage.
#' @param n_features See Usage.
#' @param n_fields See Usage.
#' @param k_dim Defaults to \code{4L}.
#' @param eta Defaults to \code{0.1}.
#' @param lam Defaults to \code{2e-05}.
#' @param epochs Defaults to \code{10L}.
#' @param seed Defaults to \code{0L}.
#' @return The value of \code{.fit_ffm}.
#' @export
morie_ffmFM <- function(rows, labels, fields, n_features, n_fields,
                        k_dim = 4L, eta = 0.1, lam = 2e-5, epochs = 10L,
                        seed = 0L) {
  .fit_ffm(rows = rows, labels = labels, fields = fields,
           n_features = n_features, n_fields = n_fields,
           k_dim = k_dim, eta = eta, lam = lam, epochs = epochs,
           seed = seed)
}
