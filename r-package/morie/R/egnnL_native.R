# E(n)-equivariant graph neural networks.
#
# Sources: Satorras, V. G., Hoogeboom, E. & Welling, M. (2021) "E(n)
# Equivariant Graph Neural Networks", Proceedings of the 38th
# International Conference on Machine Learning (ICML 2021), PMLR 139,
# 9323-9332, arXiv:2102.09844. Sec. 3 (the EGCL of eqs. (3)-(6), with
# C = 1/(M-1); the statement that eq. (4) is the main difference from
# standard GNNs and the reason equivariances 1 and 2 are preserved).
# Sec. 3.1 (the equivariance condition Qx + g; that m_ij is E(n)
# invariant because it depends on positions only through squared
# distances; that the weighted sum of differences transforms as a
# type-1 vector; and the inductive argument for composed layers). Sec.
# 3.2 (the momentum variant replacing eq. (4)). Thomas, N., Smidt, T.,
# Kearnes, S., Yang, L., Li, L., Kohlhoff, K. & Riley, P. (2018)
# "Tensor Field Networks: Rotation- and Translation-Equivariant
# Neural Networks for 3D Point Clouds", arXiv:1802.08219. The
# higher-order-representation approach this avoids.
#
# Native implementation mirroring Python morie.fn.egnnL exactly: the
# same four equations, the same equivariance check, the same
# RichResult-style payload as a named list.

.EGNNL_MODES <- c("position", "momentum")

.sqdist <- function(a, b) {
  sum((a - b) ^ 2)
}

edge_message <- function(h_i, h_j, x_i, x_j, phi_e, a_ij = NULL) {
  phi_e(as.numeric(h_i), as.numeric(h_j),
        .sqdist(as.numeric(x_i), as.numeric(x_j)),
        a_ij)
}

coord_update <- function(X, M, phi_x, C = NULL) {
  X <- as.matrix(X)
  n <- nrow(X)
  if (n < 2L)
    stop("egnnL: need at least 2 particles")
  c_val <- if (is.null(C)) 1 / (n - 1) else as.numeric(C)
  out <- matrix(0, nrow = n, ncol = ncol(X))
  for (i in seq_len(n)) {
    acc <- as.numeric(X[i, ])
    for (j in seq_len(n)) {
      if (j == i) next
      w <- as.numeric(phi_x(M[[i]][[j]]))
      acc <- acc + c_val * (X[i, ] - X[j, ]) * w
    }
    out[i, ] <- acc
  }
  lapply(seq_len(nrow(out)), function(i) as.numeric(out[i, ]))
}

egcl <- function(H, X, phi_e, phi_x, phi_h, A = NULL, C = NULL,
                 V = NULL, mode = "position", phi_v = NULL,
                 dt = 1.0) {
  if (!(mode %in% .EGNNL_MODES))
    stop("egnnL: mode must be one of ",
         paste(.EGNNL_MODES, collapse = ", "), ", got ", mode)
  H <- as.matrix(H)
  X <- as.matrix(X)
  n <- nrow(H)
  M <- replicate(n, replicate(n, NULL), simplify = FALSE)
  for (i in seq_len(n)) {
    for (j in seq_len(n)) {
      if (i != j) {
        a <- if (is.null(A)) NULL else A[[paste(i, j, sep = ",")]]
        M[[i]][[j]] <- edge_message(H[i, ], H[j, ], X[i, ], X[j, ],
                                    phi_e, a)
      }
    }
  }
  if (mode == "position") {
    Xn <- coord_update(X, M, phi_x, C)
    Vn <- V
  } else {
    if (is.null(V) || is.null(phi_v))
      stop("egnnL: the momentum variant needs V and phi_v")
    c_val <- if (is.null(C)) 1 / (n - 1) else as.numeric(C)
    Vn <- vector("list", n)
    for (i in seq_len(n)) {
      acc <- as.numeric(phi_v(H[i, ])) * as.numeric(V[[i]])
      for (j in seq_len(n)) {
        if (j == i) next
        w <- as.numeric(phi_x(M[[i]][[j]]))
        acc <- acc + c_val * (X[i, ] - X[j, ]) * w
      }
      Vn[[i]] <- acc
    }
    Xn <- lapply(seq_len(n), function(i)
      as.numeric(X[i, ]) + as.numeric(dt) * as.numeric(Vn[[i]]))
  }
  Hn <- vector("list", n)
  for (i in seq_len(n)) {
    mi <- NULL
    for (j in seq_len(n)) {
      if (j == i) next
      if (is.null(mi)) {
        mi <- as.numeric(M[[i]][[j]])
      } else {
        mi <- mi + as.numeric(M[[i]][[j]])
      }
    }
    Hn[[i]] <- phi_h(as.numeric(H[i, ]), mi)
  }
  list(H = Hn, X = Xn, V = Vn, messages = M)
}

run_egnn <- function(H, X, layers, phi_e, phi_x, phi_h, A = NULL,
                     C = NULL) {
  h <- apply(as.matrix(H), 1, as.numeric)
  x <- apply(as.matrix(X), 1, as.numeric)
  if (is.null(dim(h))) h <- matrix(h, ncol = 1)
  if (is.null(dim(x))) x <- matrix(x, ncol = 1)
  for (k in seq_len(as.integer(layers))) {
    r <- egcl(h, x, phi_e, phi_x, phi_h, A, C)
    h <- do.call(rbind, r$H)
    x <- do.call(rbind, r$X)
  }
  H_out <- lapply(seq_len(nrow(h)), function(i) as.numeric(h[i, ]))
  X_out <- lapply(seq_len(nrow(x)), function(i) as.numeric(x[i, ]))
  list(estimate = list(H_out, X_out), H = H_out, X = X_out,
       layers = as.integer(layers),
       method = paste("EGNN; Satorras, Hoogeboom & Welling (2021)",
                      "eqs. (3)-(6)"),
       note = "h is E(n) INVARIANT, x is E(n) EQUIVARIANT")
}

morie_egnnL_equivariance_error <- function(H, X, phi_e, phi_x, phi_h, Q, g,
                               layers = 2, C = NULL) {
  X <- as.matrix(X)
  n <- nrow(X)
  d <- ncol(X)
  Q <- as.matrix(Q)
  g <- as.numeric(g)
  base <- run_egnn(H, X, layers, phi_e, phi_x, phi_h, C = C)
  Xt <- t(apply(X, 1, function(r) as.numeric(Q %*% r) + g))
  if (is.null(dim(Xt))) Xt <- matrix(Xt, ncol = d)
  other <- run_egnn(H, Xt, layers, phi_e, phi_x, phi_h, C = C)
  base_X <- do.call(rbind, base$X)
  want <- t(apply(base_X, 1, function(r) as.numeric(Q %*% r) + g))
  other_X <- do.call(rbind, other$X)
  ex <- max(abs(other_X - want))
  base_H <- do.call(rbind, base$H)
  other_H <- do.call(rbind, other$H)
  eh <- max(abs(other_H - base_H))
  list(coordinate_error = ex, feature_error = eh,
       equivariant = ex < 1e-9, invariant = eh < 1e-9,
       note = "x must transform WITH Q and g; h must not move at all")
}

.egnnL_cheatsheet <- function() {
  paste("egnnL: equivariance to translation, rotation and reflection",
        "WITHOUT spherical harmonics. m_ij depends on position only",
        "through ||x_i - x_j||^2, so it is invariant; x_i <- x_i + C",
        "sum_j (x_i - x_j) phi_x(m_ij) adds a weighted sum of",
        "RELATIVE DIFFERENCES, which transforms as a vector. That",
        "one equation is the entire difference from a standard GNN.",
        "C = 1/(M-1). Composition preserves both properties by",
        "induction. A momentum variant replaces eq. (4) when",
        "velocity matters.")
}

morie_egnnL <- function(H, X, layers, phi_e, phi_x, phi_h, A = NULL,
                        C = NULL) {
  run_egnn(H, X, layers, phi_e, phi_x, phi_h, A, C)
}

equivariantgnn <- function(H, X, layers, phi_e, phi_x, phi_h,
                           A = NULL, C = NULL) {
  run_egnn(H, X, layers, phi_e, phi_x, phi_h, A, C)
}

egnn_layer <- function(H, X, layers, phi_e, phi_x, phi_h, A = NULL,
                       C = NULL) {
  run_egnn(H, X, layers, phi_e, phi_x, phi_h, A, C)
}

egnnlayer <- function(H, X, layers, phi_e, phi_x, phi_h, A = NULL,
                      C = NULL) {
  run_egnn(H, X, layers, phi_e, phi_x, phi_h, A, C)
}
