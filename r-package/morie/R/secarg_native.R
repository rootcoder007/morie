# Argon2: make the attacker pay for memory, not just for time.
# Sources: Biryukov, A., Dinu, D., Khovratovich, D. & Josefsson, S.
# (2021) "Argon2 Memory-Hard Function for Password Hashing and
# Proof-of-Work Applications", RFC 9106, doi:10.17487/RFC9106; the
# design and trade-off analysis in Biryukov, Dinu & Khovratovich
# (2016) "Argon2", EuroS&P 292-302, doi:10.1109/EuroSP.2016.31; and
# the BLAKE2 hash in Saarinen & Aumasson (2015) RFC 7693.
#
# Native R arm mirroring the Python arm exactly: the same block
# permutation P applied to the 16-word rows and to the indexed
# columns, the same pre-hashing digest H_0 over every parameter so
# changing the variant, the memory, the passes or the lanes changes
# the tag, and the same data-independent addressing rule for
# Argon2i and for the first half of the first pass of Argon2id.

.MASK64 <- 0xffffffffffffffff
.MASK32 <- 0xffffffff
.BLOCK <- 1024
.SL <- 4
.TYPES <- c(argon2d = 0L, argon2i = 1L, argon2id = 2L)
.VERSION <- 0x13

# little-endian 32-bit pack of an integer (returns raw bytes)
#' Little-endian 32-bit pack of an integer (returns raw bytes)
#'
#' A step of the secarg_native implementation. Called by \code{morie_secarg_prehash},
#' \code{morie_secarg_variable_hash}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param n Coerced to numeric by the body, with \code{as.numeric}.
#' @return The value of \code{as.raw}.
#' @export
#' @examples
#' res <- .le32(n = 3L)
#' res
.le32 <- function(n) {
  # bitwAnd(n, 0xffffffff) is NA in R: 4294967295 is past .Machine
  # integer.max, so the whole prefix came back as zero bytes and every
  # digest in this file was wrong. The Python arm is
  # int(n).to_bytes(4, "little"); do that arithmetically.
  v <- as.numeric(n)
  if (!is.finite(v) || v < 0 || v >= 4294967296)
    stop(sprintf("secarg: %s does not fit in an unsigned 32-bit word",
                 format(n)))
  as.raw(c(v %% 256,
           (v %/% 256) %% 256,
           (v %/% 65536) %% 256,
           (v %/% 16777216) %% 256))
}

#' .le64
#'
#' A step of the secarg_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param n Numeric; combined arithmetically in the body.
#' @return The value of \code{as.raw}.
#' @export
#' @examples
#' res <- .le64(n = 3L)
#' res
.le64 <- function(n) {
  n <- as.numeric(n)
  v <- n %% .MASK64
  v <- if (v < 0) v + .MASK64 else v
  b <- integer(8)
  for (k in 0:7) b[k + 1L] <- bitwAnd(bitwShiftR(v, 8L * k), 0xff)
  as.raw(b)
}

#' @param x See Usage.
#' @keywords internal
.secarg_as_bytes <- function(x) {
  if (is.raw(x)) return(x)
  if (is.character(x)) return(charToRaw(paste(x, collapse = "")))
  if (is.null(x)) return(raw(0))
  stop("expected raw, character or NULL")
}

#' Argon2's variable-length hash H' (BLAKE2b stretched past 64 bytes)
#'
#' @param data Bytes.
#' @param length Desired output length in bytes.
#' @return Raw bytes of that length.
#' @references RFC 9106 Sec. 3.3.
#' @export
morie_secarg_variable_hash <- function(data, length) {
  T <- as.integer(length)
  if (T < 1L) stop("secarg: the output length must be positive")
  a <- .secarg_as_bytes(data)
  if (T <= 64L) return(.morie_blake2b_impl(c(.le32(T), a), T))
  r <- -(-T %/% 32L) - 2L
  out <- raw(0)
  v <- .morie_blake2b_impl(c(.le32(T), a), 64L)
  out <- c(out, v[1:32])
  for (kk in seq_len(r - 1L)) {
    v <- .morie_blake2b_impl(v, 64L)
    out <- c(out, v[1:32])
  }
  v <- .morie_blake2b_impl(v, T - 32L * r)
  out <- c(out, v)
  out[seq_len(T)]
}

#' Pre-hashing digest H_0 over every parameter
#'
#' @param password See Usage.
#' @param salt See Usage.
#' @param parallelism See Usage.
#' @param tag_length See Usage.
#' @param memory See Usage.
#' @param passes See Usage.
#' @param variant See Usage.
#' @param secret See Usage.
#' @param associated See Usage.
#' @param version See Usage.
#' @return Raw 64-byte digest.
#' @references RFC 9106 Sec. 3.1.
#' @export
morie_secarg_prehash <- function(password, salt, parallelism, tag_length, memory,
                    passes, variant = "argon2id", secret = raw(0),
                    associated = raw(0), version = .VERSION) {
  y <- unname(.TYPES[tolower(as.character(variant))])
  if (is.na(y)) {
    stop("secarg: variant must be one of ",
         paste(names(.TYPES), collapse = ", "), ", got ",
         deparse(variant))
  }
  P <- .secarg_as_bytes(password)
  S <- .secarg_as_bytes(salt)
  K <- .secarg_as_bytes(secret)
  X <- .secarg_as_bytes(associated)
  if (length(S) < 8L) {
    stop("secarg: the salt must be at least 8 bytes ",
         "(the RFC recommends 16), got ", length(S))
  }
  buf <- c(.le32(parallelism), .le32(tag_length), .le32(memory),
           .le32(passes), .le32(version), .le32(y),
           .le32(length(P)), P, .le32(length(S)), S,
           .le32(length(K)), K, .le32(length(X)), X)
  .morie_blake2b_impl(buf, 64L)
}

# G function's 8-word permutation on a 16-word vector.
#' G function\'s 8-word permutation on a 16-word vector
#'
#' A step of the secarg_native implementation. Called by \code{.P_step}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param v A vector; indexed elementwise.
#' @param a See Usage.
#' @param b See Usage.
#' @param c See Usage.
#' @param d See Usage.
#' @return The value of \code{v}, as built in the body.
#' @export
.gb <- function(v, a, b, c, d) {
  va <- v[a]
  vb <- v[b]
  vc <- v[c]
  vd <- v[d]
  # mult (low-32) * 2 * (high-32) emulated by bitwAnd + bitwShiftR pieces
  va_l <- bitwAnd(va, .MASK32)
  va_h <- bitwShiftR(va, 32)
  vb_l <- bitwAnd(vb, .MASK32)
  vb_h <- bitwShiftR(vb, 32)
  vc_l <- bitwAnd(vc, .MASK32)
  vc_h <- bitwShiftR(vc, 32)
  vd_l <- bitwAnd(vd, .MASK32)
  vd_h <- bitwShiftR(vd, 32)

  # 64-bit product
  mul6464 <- function(x, y) {
    a_lo <- bitwAnd(x, .MASK32)
    a_hi <- bitwShiftR(x, 32)
    b_lo <- bitwAnd(y, .MASK32)
    b_hi <- bitwShiftR(y, 32)
    p1 <- bitwShiftL(a_lo * b_lo, 0)
    p2 <- bitwShiftL(a_lo * b_hi, 32)
    p3 <- bitwShiftL(a_hi * b_lo, 32)
    p4 <- bitwShiftL(a_hi * b_hi, 64)
    bitwAnd(p1 + p2 + p3 + p4, .MASK64)
  }

  # v[a] = v[a] + v[b] + 2*low32(v[a])*low32(v[b])  (mod 2^64)
  v[a] <- bitwAnd(va + vb + 2L * va_l * vb_l, .MASK64)
  # v[d] = ROR32(v[d] xor v[a])  ==  ((v[d]^v[a])>>32)|((v[d]^v[a])<<32) & M64
  t <- bitwXor(v[d], v[a])
  v[d] <- bitwAnd(bitwOr(bitwShiftR(t, 32),
                         bitwAnd(bitwShiftL(t, 32), .MASK64)),
                  .MASK64)
  # v[c] = v[c] + v[d] + 2*low32(v[c])*low32(v[d])
  vc <- v[c]
  vd <- v[d]
  v[c] <- bitwAnd(vc + vd + 2L * bitwAnd(vc, .MASK32) * bitwAnd(vd, .MASK32),
                  .MASK64)
  # v[b] = ROR24(v[b] xor v[c])
  vb <- v[b]
  vc <- v[c]
  x <- bitwXor(vb, vc)
  v[b] <- bitwAnd(bitwOr(bitwShiftR(x, 24),
                         bitwAnd(bitwShiftL(x, 40), .MASK64)),
                  .MASK64)
  # v[a] = v[a] + v[b] + 2*low32(v[a])*low32(v[b])
  va <- v[a]
  vb <- v[b]
  v[a] <- bitwAnd(va + vb + 2L * bitwAnd(va, .MASK32) * bitwAnd(vb, .MASK32),
                  .MASK64)
  # v[d] = ROR16(v[d] xor v[a])
  vd <- v[d]
  va <- v[a]
  x <- bitwXor(vd, va)
  v[d] <- bitwAnd(bitwOr(bitwShiftR(x, 16),
                         bitwAnd(bitwShiftL(x, 48), .MASK64)),
                  .MASK64)
  # v[c] = v[c] + v[d] + 2*low32(v[c])*low32(v[d])
  vc <- v[c]
  vd <- v[d]
  v[c] <- bitwAnd(vc + vd + 2L * bitwAnd(vc, .MASK32) * bitwAnd(vd, .MASK32),
                  .MASK64)
  # v[b] = ROR63(v[b] xor v[c])
  vb <- v[b]
  vc <- v[c]
  x <- bitwXor(vb, vc)
  v[b] <- bitwAnd(bitwOr(bitwShiftR(x, 63),
                         bitwAnd(bitwShiftL(x, 1), .MASK64)),
                  .MASK64)
  v
}

#' .P_step
#'
#' A step of the secarg_native implementation. Called by \code{morie_secarg_compress}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param v Passed to \code{.gb}.
#' @return The value of \code{v}, as built in the body.
#' @export
.P_step <- function(v) {
  v <- .gb(v, 1L, 5L, 9L, 13L)
  v <- .gb(v, 2L, 6L, 10L, 14L)
  v <- .gb(v, 3L, 7L, 11L, 15L)
  v <- .gb(v, 4L, 8L, 12L, 16L)
  v <- .gb(v, 1L, 6L, 11L, 16L)
  v <- .gb(v, 2L, 7L, 12L, 13L)
  v <- .gb(v, 3L, 8L, 9L, 14L)
  v <- .gb(v, 4L, 5L, 10L, 15L)
  v
}

#' Compression function G(X, Y) = R xor Q
#' @param X See Usage.
#' @param Y See Usage.
#' @export
morie_secarg_compress <- function(X, Y) {
  R <- mapply(bitwXor, X, Y, SIMPLIFY = FALSE)
  Q <- R
  for (i in 0:7) {
    row <- Q[(16L * i + 1L):(16L * i + 16L)]
    row <- .P_step(row)
    Q[(16L * i + 1L):(16L * i + 16L)] <- row
  }
  idx <- integer(16)
  for (i in 0:7) {
    idx[2L * i + 1L] <- 16L * i + 2L * 0L + 1L
    idx[2L * i + 2L] <- 16L * i + 2L * 0L + 2L
  }
  # The Python actually builds idx with 2*j for j in 0..7 per row.
  idx <- integer(16)
  pos <- 1L
  for (j in 0:7) for (i in 0:7) {
    idx[pos] <- 16L * i + 2L * j + 1L
    pos <- pos + 1L
    idx[pos] <- 16L * i + 2L * j + 2L
    pos <- pos + 1L
  }
  col <- Q[idx]
  col <- .P_step(col)
  Q[idx] <- col
  mapply(bitwXor, Q, R, SIMPLIFY = FALSE)
}

#' .to_words
#'
#' A step of the secarg_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param bs A vector; its length is taken and its elements indexed.
#' @return The value of \code{out}, as built in the body.
#' @export
.to_words <- function(bs) {
  n <- length(bs) %/% 8L
  out <- integer(n)
  for (i in seq_len(n)) {
    v <- 0L
    chunk <- bs[(8L * (i - 1L) + 1L):(8L * i)]
    for (k in 0:7) {
      v <- v + as.integer(chunk[k + 1L]) * bitwShiftL(1L, 8L * k)
    }
    out[i] <- v
  }
  out
}

#' .to_bytes
#'
#' A step of the secarg_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param ws Iterated over elementwise, with \code{lapply}.
#' @return The value of \code{do.call}.
#' @export
.to_bytes <- function(ws) {
  do.call(c, lapply(ws, .le64))
}

#' .addresses
#'
#' A step of the secarg_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param pass_no See Usage.
#' @param lane See Usage.
#' @param slice_no See Usage.
#' @param m_prime See Usage.
#' @param passes See Usage.
#' @param y See Usage.
#' @param counter See Usage.
#' @return The value of \code{morie_secarg_compress}.
#' @export
.addresses <- function(pass_no, lane, slice_no, m_prime, passes, y,
                       counter) {
  zero <- as.list(rep(0L, 128L))
  inp <- as.list(rep(0L, 128L))
  inp[[1L]] <- pass_no
  inp[[2L]] <- lane
  inp[[3L]] <- slice_no
  inp[[4L]] <- m_prime
  inp[[5L]] <- passes
  inp[[6L]] <- y
  inp[[7L]] <- counter
  morie_secarg_compress(zero, morie_secarg_compress(zero, inp))
}

#' Argon2 password hash (RFC 9106)
#' @param password See Usage.
#' @param salt See Usage.
#' @param memory See Usage.
#' @param passes See Usage.
#' @param parallelism See Usage.
#' @param tag_length See Usage.
#' @param variant See Usage.
#' @param secret See Usage.
#' @param associated See Usage.
#' @export
morie_secarg_argon2 <- function(password, salt, memory = 32, passes = 3,
                                parallelism = 4, tag_length = 32,
                                variant = "argon2id", secret = NULL,
                                associated = NULL) {
  tag <- .morie_argon2_impl(password, salt, as.integer(memory),
                           as.integer(passes), as.integer(parallelism),
                           as.integer(tag_length), as.character(variant),
                           secret, associated)
  p <- as.integer(parallelism)
  m <- as.integer(memory)
  m_prime <- (m %/% (4L * p)) * (4L * p)
  y <- switch(as.character(variant), argon2d = 0L, argon2i = 1L,
              argon2id = 2L)
  hex <- paste(sprintf("%02x", as.integer(tag)), collapse = "")
  list(estimate = hex, tag = tag, tag_hex = hex,
       variant = as.character(variant), memory_kib = m,
       memory_used_kib = m_prime, passes = as.integer(passes),
       parallelism = p, version = 19L,
       data_independent_first_half = (y == 2L),
       method = paste0("Argon2 v1.3; Biryukov, Dinu, Khovratovich & ",
                       "Josefsson (2021) RFC 9106"),
       note = paste0("a tag is only comparable against another computed ",
                     "under the SAME parameters, which is why they are ",
                     "returned with it"))
}

#' .secarg_hexlify
#'
#' A step of the secarg_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param bs Coerced to integer by the body, with \code{as.integer}.
#' @return A character value.
#' @export
.secarg_hexlify <- function(bs) {
  paste(format(as.hexmode(as.integer(bs)), width = 2,
               upper.case = TRUE), collapse = "")
}

#' Recommended Argon2 configurations from RFC 9106 Sec. 4
#' @param profile See Usage.
#' @export
morie_secarg_parameter_advice <- function(profile = "first") {
  rec <- list(
    first = list(variant = "argon2id", memory = 2L * 1024L * 1024L,
                 passes = 1L, parallelism = 4L, tag_length = 32L,
                 salt_bytes = 16L,
                 note = paste("RFC 9106 Sec. 4 first recommended ",
                              "option: 2 GiB, t = 1, p = 4")),
    second = list(variant = "argon2id", memory = 64L * 1024L,
                  passes = 3L, parallelism = 4L, tag_length = 32L,
                  salt_bytes = 16L,
                  note = paste("RFC 9106 Sec. 4 second option for ",
                               "memory-constrained environments: 64 ",
                               "MiB, t = 3, p = 4"))
  )
  if (!profile %in% names(rec)) {
    stop("secarg: profile must be 'first' or 'second', got ",
         deparse(profile))
  }
  out <- rec[[profile]]
  out$memory_gib <- out$memory / (1024.0 * 1024.0)
  out$warning <- paste("lowering memory in favour of more passes ",
                       "weakens time-space trade-off resistance")
  out
}

# house entry point: the package exports one morie_<module>
morie_secarg <- morie_secarg_argon2
