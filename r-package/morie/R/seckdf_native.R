# HKDF: extract entropy, then expand it -- two steps, on purpose.
# Sources: Krawczyk, H. & Eronen, P. (2010) "HMAC-based Extract-and-
# Expand Key Derivation Function (HKDF)", RFC 5869,
# doi:10.17487/RFC5869 (the extract/expand split, the 255*HashLen
# cap, the role of the salt as HMAC key with the IKM as the
# message); Krawczyk, H. (2010) "Cryptographic Extraction and Key
# Derivation: The HKDF Scheme", CRYPTO 2010 LNCS 6223, 631-648
# (the analysis behind the split); and NIST (2008) FIPS PUB 198-1
# HMAC, the MAC underneath.
#
# Native R arm mirroring the Python arm exactly: the same extract
# with the salt as HMAC key and the IKM as the message, the same
# counter-mode expansion with a single-octet counter, and the same
# 255*HashLen cap and PRK-length check on the same inputs.

.HASH_LEN <- 32L
.MAX_BLOCKS <- 255L

#' .seckdf_as_bytes
#'
#' A step of the seckdf_native implementation. Called by \code{derive_context_keys}, \code{expand}, \code{extract} and 1 others in the module.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param x Optional; may be \code{NULL}. Character; the body checks with \code{is.character}.
#' @return Nothing; this branch always raises.
#' @export
.seckdf_as_bytes <- function(x) {
  if (is.raw(x)) return(x)
  if (is.character(x)) return(charToRaw(paste(x, collapse = "")))
  if (is.null(x)) return(raw(0))
  stop("expected raw, character or NULL")
}

#' .seckdf_hexlify
#'
#' A step of the seckdf_native implementation. Called by \code{hkdf}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param bs Coerced to integer by the body, with \code{as.integer}.
#' @return A character value.
#' @export
.seckdf_hexlify <- function(bs) {
  paste(format(as.hexmode(as.integer(bs)), width = 2,
               upper.case = TRUE), collapse = "")
}

#' HKDF Extract: PRK = HMAC(salt, IKM)
#' @param ikm See Usage.
#' @param salt See Usage.
#' @export
extract <- function(ikm, salt = NULL) {
  s <- if (is.null(salt)) raw(.HASH_LEN) else .seckdf_as_bytes(salt)
  list(prk = .morie_hmac_sha256_impl(s, ikm),
       salt_supplied = !is.null(salt),
       note = paste("the salt is the HMAC KEY; the IKM is the ",
                    "message"))
}

#' HKDF Expand: counter-mode OKM
#' @param prk See Usage.
#' @param info See Usage.
#' @param length See Usage.
#' @export
expand <- function(prk, info = raw(0), length = 32L) {
  L <- as.integer(length)
  if (L < 1L) stop("seckdf: the output length must be positive")
  if (L > .MAX_BLOCKS * .HASH_LEN) {
    stop("seckdf: L = ", L, " exceeds 255*HashLen = ",
         .MAX_BLOCKS * .HASH_LEN,
         "; the counter is a single octet, so this cannot be ",
         "satisfied")
  }
  p <- .seckdf_as_bytes(prk)
  if (length(p) < .HASH_LEN) {
    stop("seckdf: the PRK is ", length(p), " bytes, shorter than ",
         "the hash length ", .HASH_LEN,
         " -- Extract was probably skipped on non-uniform input")
  }
  inf <- .seckdf_as_bytes(info)
  out <- raw(0)
  t <- raw(0)
  i <- 1L
  while (length(out) < L) {
    t <- .morie_hmac_sha256_impl(p, c(t, inf, as.raw(i)))
    out <- c(out, t)
    i <- i + 1L
  }
  list(okm = out[seq_len(L)], blocks = i - 1L, length = L)
}

#' HKDF: extract-then-expand, or expand-only on uniform input
#' @param ikm See Usage.
#' @param salt See Usage.
#' @param info See Usage.
#' @param length See Usage.
#' @param skip_extract See Usage.
#' @export
hkdf <- function(ikm, salt = NULL, info = raw(0), length = 32L,
                 skip_extract = FALSE) {
  if (isTRUE(skip_extract)) {
    prk <- .seckdf_as_bytes(ikm)
    salted <- FALSE
  } else {
    e <- extract(ikm, salt)
    prk <- e$prk
    salted <- e$salt_supplied
  }
  r <- expand(prk, info, length)
  list(estimate = .seckdf_hexlify(r$okm), okm = r$okm,
       okm_hex = .seckdf_hexlify(r$okm), prk = prk,
       prk_hex = .seckdf_hexlify(prk), length = r$length,
       blocks = r$blocks, salt_supplied = salted,
       extract_skipped = isTRUE(skip_extract),
       method = "HKDF-SHA256; Krawczyk & Eronen (2010) RFC 5869",
       note = paste("info binds the output to a context, so one PRK ",
                    "gives independent keys for independent purposes"))
}

#' Derive one key per context, all independent
#' @param ikm See Usage.
#' @param contexts See Usage.
#' @param salt See Usage.
#' @param length See Usage.
#' @export
derive_context_keys <- function(ikm, contexts, salt = NULL,
                                length = 32L) {
  e <- extract(ikm, salt)
  keys <- list()
  for (c in contexts) {
    keys[[c]] <- expand(e$prk, .seckdf_as_bytes(c), length)$okm
  }
  hexed <- lapply(keys, .seckdf_hexlify)
  distinct <- length(unique(unlist(hexed))) == length(hexed)
  list(keys = keys, hex = hexed, prk = e$prk,
       all_distinct = distinct,
       note = paste("same PRK, different info, unrelated outputs"))
}

# house entry point: the package exports one morie_<module>
morie_seckdf <- hkdf
