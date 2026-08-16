# morie.fn -- function file (rootcoder007/morie)
# PaiNN: equivariant message passing for tensorial properties.
#
# Sources: Schutt, K. T., Unke, O. T. & Gastegger, M. (2021)
# "Equivariant message passing for the prediction of tensorial
# properties and molecular spectra", ICML 2021, PMLR 139, 9377-9388,
# arXiv:2102.03150. Message passing networks scale readily to large
# training sets but have proven less data efficient than kernel
# methods; the identification of the limitations of invariant
# representations as a major reason; the extension of message
# passing to rotationally equivariant representations; the
# polarizable atom interaction neural network improving on common
# molecule benchmarks while reducing model size and inference
# time; and the use of equivariant atomwise representations for
# tensorial properties and molecular spectra, with speedups of 4-5
# orders of magnitude over the electronic structure reference.
#
# Schutt, K. T., Kindermans, P.-J., Sauceda, H. E., Chmiela, S.,
# Tkatchenko, A. & Muller, K.-R. (2017) "SchNet", NeurIPS 2017,
# arXiv:1706.08566. The invariant predecessor.
#
# Satorras, V. G., Hoogeboom, E. & Welling, M. (2021) "E(n) Equivariant
# Graph Neural Networks", ICML 2021, PMLR 139, 9323-9332,
# arXiv:2102.09844.

.painn_EPS <- 1e-12

# ||v|| -- an invariant built from an equivariant.
#' ||v|| -- an invariant built from an equivariant
#'
#' A step of the painn_native implementation. Called by \code{gated_update}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param v A matrix; passed to \code{as.matrix}.
#' @return One of two values, depending on the branch taken.
#' @export
vector_norm <- function(v) {
  a <- as.matrix(v)
  if (ncol(a) == 3L) {
    # v is given as a 3xF matrix (d=3, f features)
    d <- nrow(a)
    f <- ncol(a)
    out <- numeric(f)
    for (k in seq_len(f))
      out[k] <- sqrt(sum(a[, k] ^ 2))
    out
  } else {
    # v is a list of length d, each element a length-f vector
    d <- length(a); f <- length(a[[1]])
    out <- numeric(f)
    for (k in seq_len(f))
      out[k] <- sqrt(sum(sapply(a, function(r) r[k]) ^ 2))
    out
  }
}

# The message: scalars from scalars, vectors from s*hat_r and s*v.
# Type is preserved by construction, which is what makes the whole
# network equivariant rather than approximately so.
#' The message: scalars from scalars, vectors from s*hat_r and s*v
#'
#' Type is preserved by construction, which is what makes the whole
#' network equivariant rather than approximately so.
#'
#' @param s_j Coerced to numeric by the body, with \code{as.numeric}.
#' @param v_j A matrix; passed to \code{as.matrix}.
#' @param r_ij Coerced to numeric by the body, with \code{as.numeric}.
#' @param phi_s Accepted by the signature and not used anywhere in the body.
#' @param phi_v Accepted by the signature and not used anywhere in the body.
#' @param W_rbf Accepted by the signature and not used anywhere in the body.
#' @return A list with \code{ds}, \code{dv}, \code{note}.
#' @export
scalar_vector_message <- function(s_j, v_j, r_ij, phi_s, phi_v,
                                  W_rbf) {
  s <- as.numeric(s_j)
  V <- as.matrix(v_j)
  r <- as.numeric(r_ij)
  d <- sqrt(sum(r * r))
  if (d <= .painn_EPS)
    stop("painn: two atoms occupy the same position")
  hat <- r / d
  w <- as.numeric(W_rbf(d))
  ds <- as.numeric(phi_s(s, w))
  dv_scale <- as.numeric(phi_v(s, w))
  F <- length(s)
  if (length(ds) != F || length(dv_scale) != 2L * F)
    stop("painn: the message networks are mis-sized (need F ",
         "scalars and 2F vector gates)")
  # v_j is laid out as F columns of 3 rows; rebuild as 3 rows of F
  if (ncol(V) == F && nrow(V) == 3L) {
    Vt <- V
  } else if (nrow(V) == F && ncol(V) == 3L) {
    Vt <- t(V)
  } else {
    Vt <- V
  }
  D <- 3L
  dv <- matrix(0, nrow = D, ncol = F)
  for (a in seq_len(D))
    for (f in seq_len(F))
      dv[a, f] <- dv_scale[f] * Vt[a, f] + dv_scale[F + f] * hat[a]
  list(ds = ds, dv = dv,
       note = paste("scalar*vector and s*r_hat give VECTORS; ",
                    "nothing mixes the types", sep = ""))
}

# The update block: v1 . v2 re-enters the scalars. That inner product
# is the only path from the vector channel back to the scalar one,
# and it is invariant -- which is why the network can use
# directional information without breaking invariance of the energy.
#' The update block: v1 . v2 re-enters the scalars. That inner product
#'
#' is the only path from the vector channel back to the scalar one, and
#' it is invariant -- which is why the network can use directional
#' information without breaking invariance of the energy.
#'
#' @param s Coerced to numeric by the body, with \code{as.numeric}.
#' @param v A matrix; passed to \code{as.matrix}.
#' @param U A matrix; passed to \code{as.matrix}.
#' @param V A matrix; passed to \code{as.matrix}.
#' @param phi Accepted by the signature and not used anywhere in the body.
#' @return A list with \code{ds}, \code{dv}, \code{scalar_from_vectors}, \code{note}.
#' @export
gated_update <- function(s, v, U, V, phi) {
  sv <- as.numeric(s)
  Vv <- as.matrix(v)
  if (nrow(Vv) != 3L) Vv <- t(Vv)
  D <- nrow(Vv); F <- ncol(Vv)
  Um <- as.matrix(U); Vm <- as.matrix(V)
  if (nrow(Um) != F) Um <- t(Um)
  if (nrow(Vm) != F) Vm <- t(Vm)
  Uv <- matrix(0, nrow = D, ncol = F)
  Vw <- matrix(0, nrow = D, ncol = F)
  for (a in seq_len(D)) {
    for (f in seq_len(F)) {
      acc1 <- 0; acc2 <- 0
      for (g in seq_len(F)) {
        acc1 <- acc1 + Um[f, g] * Vv[a, g]
        acc2 <- acc2 + Vm[f, g] * Vv[a, g]
      }
      Uv[a, f] <- acc1; Vw[a, f] <- acc2
    }
  }
  dot <- numeric(F)
  for (f in seq_len(F))
    dot[f] <- sum(Uv[, f] * Vw[, f])
  nrm <- vector_norm(Vw)
  out <- phi(sv, dot, nrm)
  ds <- as.numeric(out$ds)
  gate <- as.numeric(out$gate)
  if (length(ds) != F || length(gate) != F)
    stop("painn: the update network is mis-sized")
  dv <- matrix(0, nrow = D, ncol = F)
  for (a in seq_len(D))
    for (f in seq_len(F))
      dv[a, f] <- gate[f] * Uv[a, f]
  list(ds = ds, dv = dv, scalar_from_vectors = dot,
       note = paste("the vector-vector inner product is the ONLY ",
                    "path back to the scalar channel, and it is ",
                    "invariant", sep = ""))
}

# mu = sum_i q_i (r_i - r_c). A tensorial (here vector) property,
# read off directly from equivariant atomwise quantities.
#' Mu = sum_i q_i (r_i - r_c). A tensorial (here vector) property,
#'
#' read off directly from equivariant atomwise quantities.
#'
#' @param charges Coerced to numeric by the body, with \code{as.numeric}.
#' @param R A matrix; passed to \code{as.matrix}.
#' @param centre Optional; may be \code{NULL}. Coerced to numeric by the body, with \code{as.numeric}.
#' @return A list with \code{dipole}, \code{magnitude}, \code{note}.
#' @export
dipole_moment <- function(charges, R, centre = NULL) {
  q <- as.numeric(charges)
  pos <- as.matrix(R)
  if (length(q) != nrow(pos))
    stop("painn: ", length(q), " charges but ", nrow(pos),
         " positions")
  d <- ncol(pos)
  ctr <- if (is.null(centre)) colMeans(pos) else as.numeric(centre)
  mu <- numeric(d)
  for (a in seq_len(d))
    for (i in seq_along(q))
      mu[a] <- mu[a] + q[i] * (pos[i, a] - ctr[a])
  list(dipole = mu, magnitude = sqrt(sum(mu * mu)),
       note = paste("a VECTOR property; an invariant network cannot ",
                    "produce one without a separate head", sep = ""))
}

# Rotate the input; scalars must not move, vectors must rotate.
# Checking only the scalars would pass a model that has silently
# lost its vector channel.
#' Rotate the input; scalars must not move, vectors must rotate
#'
#' Checking only the scalars would pass a model that has silently lost
#' its vector channel.
#'
#' @param model Accepted by the signature and not used anywhere in the body.
#' @param s See Usage.
#' @param v A matrix; passed to \code{as.matrix}.
#' @param R A matrix; passed to \code{as.matrix}.
#' @param Q A matrix; passed to \code{as.matrix}.
#' @param tol Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{1e-09}.
#' @return A list with \code{scalar_error}, \code{vector_error}, \code{scalars_invariant}, \code{vectors_equivariant}, \code{note}.
#' @export
morie_painn_equivariance_error <- function(model, s, v, R, Q, tol = 1e-9) {
  pos <- as.matrix(R)
  Qm <- as.matrix(Q)
  d <- ncol(pos)
  rot_R <- matrix(0, nrow = nrow(pos), ncol = d)
  for (i in seq_len(nrow(pos)))
    for (a in seq_len(d))
      for (b in seq_len(d))
        rot_R[i, a] <- rot_R[i, a] + Qm[a, b] * pos[i, b]
  V <- as.matrix(v)
  if (nrow(V) != d) V <- t(V)
  rot_v <- matrix(0, nrow = d, ncol = ncol(V))
  for (a in seq_len(d))
    for (f in seq_len(ncol(V)))
      for (b in seq_len(d))
        rot_v[a, f] <- rot_v[a, f] + Qm[a, b] * V[b, f]
  base <- model(s, V, pos)
  other <- model(s, rot_v, rot_R)
  se <- max(abs(as.numeric(base$s) - as.numeric(other$s)))
  want <- matrix(0, nrow = d, ncol = length(base$v[[1]]))
  if (is.list(base$v)) {
    # base$v is a list of 3 elements, each a length-F vector
    F <- length(base$v[[1]])
    for (a in seq_len(d)) for (f in seq_len(F)) {
      acc <- 0
      for (b in seq_len(d)) acc <- acc + Qm[a, b] * base$v[[b]][f]
      want[a, f] <- acc
    }
    other_v <- matrix(0, nrow = d, ncol = F)
    for (a in seq_len(d)) for (f in seq_len(F))
      other_v[a, f] <- other$v[[a]][f]
  } else {
    V2 <- as.matrix(base$v)
    if (nrow(V2) != d) V2 <- t(V2)
    F <- ncol(V2)
    for (a in seq_len(d)) for (f in seq_len(F)) {
      acc <- 0
      for (b in seq_len(d)) acc <- acc + Qm[a, b] * V2[b, f]
      want[a, f] <- acc
    }
    other_v <- as.matrix(other$v)
    if (nrow(other_v) != d) other_v <- t(other_v)
  }
  ve <- max(abs(other_v - want))
  list(scalar_error = se, vector_error = ve,
       scalars_invariant = se < as.numeric(tol),
       vectors_equivariant = ve < as.numeric(tol),
       note = paste("both must hold; checking only the scalars ",
                    "passes a model that has lost its vectors",
                    sep = ""))
}

#' .painn_cheatsheet
#'
#' A step of the painn_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @return A character value.
#' @export
.painn_cheatsheet <- function() {
  paste("painn: message passing was LESS DATA EFFICIENT than ",
        "kernel methods, and the diagnosis is INVARIANT ",
        "representations -- a network of scalars can only combine ",
        "distances and cannot emit a tensor at all. Carry BOTH a ",
        "scalar and a VECTOR feature per atom and preserve type: ",
        "s*s and ||v|| give scalars, s*v and s*r_hat give vectors, ",
        "and v1.v2 is the ONLY route back from vectors to scalars ",
        "-- invariant, so the energy stays invariant while ",
        "direction is used. Tensorial properties are read off ",
        "directly, and the model is SMALLER, not larger.", sep = "")
}

morie_painn <- gated_update
