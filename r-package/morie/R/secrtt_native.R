# Envelope encryption: a data key per row, a key-encrypting key
# above it.
# Sources: NIST (2020) SP 800-57 Part 1 Rev. 5 (key hierarchy, KEKs
# wrapping DEKs, cryptoperiods); Housley, R. (2009) RFC 5652 CMS
# (enveloped-data structure); Nir, Y. & Langley, A. (2018) RFC 8439
# (the AEAD used for both wrap and record); and Krawczyk, H. &
# Eronen, P. (2010) RFC 5869 (per-record key derivation).
#
# Native R arm mirroring the Python arm exactly: the same per-record
# HKDF derivation, the same AEAD wrap/unwrap with the KEK id
# authenticated into the AAD so a wrapped DEK cannot be replayed
# under a different KEK, and the same cost model for KEK rotation
# (cheap, zero record ciphertext touched) vs DEK rotation (one
# record rewritten).

#' .secrtt_as_bytes
#'
#' A step of the secrtt_native implementation. Called by \code{generate_dek}, \code{open_record}, \code{rotate_dek} and 3 others in the module.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param x Optional; may be \code{NULL}. Character; the body checks with \code{is.character}.
#' @return Nothing; this branch always raises.
#' @export
.secrtt_as_bytes <- function(x) {
  if (is.raw(x)) return(x)
  if (is.character(x)) return(charToRaw(paste(x, collapse = "")))
  if (is.null(x)) return(raw(0))
  stop("expected raw, character or NULL")
}

#' Derive a distinct DEK per record
#' @export
.secrtt_hex <- function(bs) paste(sprintf("%02x", as.integer(bs)),
                                 collapse = "")

.secrtt_aead_encrypt <- function(key, nonce, plaintext, aad = raw(0)) {
  r <- morie_crypto_chacha20_poly1305_encrypt(as.raw(key), as.raw(nonce),
                                              as.raw(plaintext), as.raw(aad))
  list(ciphertext = r$ct, tag = r$tag,
       ciphertext_hex = .secrtt_hex(r$ct))
}

.secrtt_aead_decrypt <- function(key, nonce, ciphertext, tag,
                                 aad = raw(0)) {
  pt <- tryCatch(
    morie_crypto_chacha20_poly1305_decrypt(as.raw(key), as.raw(nonce),
      c(as.raw(ciphertext), as.raw(tag)), as.raw(aad)),
    error = function(e) NULL)
  list(valid = !is.null(pt),
       plaintext = if (is.null(pt)) raw(0) else pt)
}

generate_dek <- function(master_seed, record_id, salt = NULL) {
  r <- hkdf(master_seed, salt,
            c(charToRaw("dek:"), .secrtt_as_bytes(record_id)), 32L)
  list(dek = r$okm, dek_hex = r$okm_hex,
       record_id = record_id,
       note = paste("per-record, so one leaked DEK exposes one ",
                    "record"))
}

#' Wrap a DEK under a KEK
#' @export
wrap_dek <- function(dek, kek, nonce, kek_id = "kek-1",
                     aad = raw(0)) {
  d <- .secrtt_as_bytes(dek)
  if (length(d) != 32L) {
    stop("secrtt: a DEK must be 32 bytes, got ", length(d))
  }
  bound <- c(.secrtt_as_bytes(aad), .secrtt_as_bytes(kek_id))
  r <- .secrtt_aead_encrypt(kek, nonce, d, bound)
  list(wrapped = r$ciphertext, tag = r$tag,
       nonce = .secrtt_as_bytes(nonce), kek_id = kek_id,
       wrapped_hex = r$ciphertext_hex,
       note = paste("the KEK id is authenticated, so a wrapped DEK ",
                    "cannot be replayed under a different KEK"))
}

#' Unwrap a DEK, optionally logging the call
#' @export
unwrap_dek <- function(wrapped, kek, audit_log = NULL) {
  r <- .secrtt_aead_decrypt(kek, wrapped$nonce, wrapped$wrapped, wrapped$tag,
                    c(.secrtt_as_bytes(wrapped$aad %||% raw(0)),
                      .secrtt_as_bytes(wrapped$kek_id)))
  if (!is.null(audit_log)) {
    audit_log[[length(audit_log) + 1L]] <- list(
      event = "unwrap", kek_id = wrapped$kek_id, ok = r$valid)
  }
  if (!r$valid) {
    stop("secrtt: the wrapped DEK failed authentication -- wrong ",
         "KEK, or it was tampered with")
  }
  list(dek = r$plaintext, kek_id = wrapped$kek_id,
       audited = !is.null(audit_log))
}

`%||%` <- function(a, b) if (is.null(a)) b else a

#' Seal a record under its DEK
#' @export
seal_record <- function(plaintext, dek, nonce, aad = raw(0)) {
  r <- .secrtt_aead_encrypt(dek, nonce, plaintext, aad)
  list(ciphertext = r$ciphertext, tag = r$tag,
       nonce = .secrtt_as_bytes(nonce), aad = .secrtt_as_bytes(aad))
}

#' Open a sealed record
#' @export
open_record <- function(sealed, dek) {
  r <- .secrtt_aead_decrypt(dek, sealed$nonce, sealed$ciphertext,
                    sealed$tag, .secrtt_as_bytes(sealed$aad %||% raw(0)))
  if (!r$valid) {
    stop("secrtt: the record failed authentication")
  }
  r$plaintext
}

#' Rotate the KEK: re-wrap every DEK, no record ciphertext touched
#' @export
rotate_kek <- function(wrapped_deks, old_kek, new_kek, new_nonces,
                       new_kek_id = "kek-2", audit_log = NULL) {
  if (length(new_nonces) != length(wrapped_deks)) {
    stop("secrtt: ", length(wrapped_deks), " wrapped DEKs but ",
         length(new_nonces), " nonces -- a nonce must never be ",
         "reused under a new KEK")
  }
  out <- list()
  for (i in seq_along(wrapped_deks)) {
    w <- wrapped_deks[[i]]
    dek <- unwrap_dek(w, old_kek, audit_log)$dek
    out[[i]] <- wrap_dek(dek, new_kek, new_nonces[[i]], new_kek_id)
  }
  list(estimate = out, wrapped = out, n = length(out),
       records_reencrypted = 0, kek_id = new_kek_id,
       method = "envelope KEK rotation; NIST SP 800-57 Part 1 Rev. 5",
       note = paste("one small re-wrap per record and zero record ",
                    "ciphertext rewritten"))
}

#' Rotate a DEK: re-encrypt one record
#' @export
rotate_dek <- function(sealed, old_dek, new_dek, new_nonce) {
  pt <- open_record(sealed, old_dek)
  list(sealed = seal_record(pt, new_dek, new_nonce,
                            .secrtt_as_bytes(sealed$aad %||% raw(0))),
       records_reencrypted = 1L,
       note = paste("rotating a DEK rewrites its record; rotating ",
                    "the KEK does not"))
}

#' Cost model: single-key vs envelope rotation
#' @export
rotation_cost <- function(n_records, mean_record_bytes,
                          dek_bytes = 32L) {
  n <- as.integer(n_records); b <- as.numeric(mean_record_bytes)
  if (n < 1L || b <= 0) {
    stop("secrtt: the record count and size must be positive")
  }
  single <- n * b
  kek <- n * as.numeric(dek_bytes)
  list(single_key_bytes = single, envelope_kek_bytes = kek,
       ratio = if (kek > 0) single / kek else Inf,
       records_touched_single = n,
       records_touched_envelope = 0L,
       note = paste("rotation that is expensive is rotation that is ",
                    "deferred"))
}

#' Crypto-shred: destroy a KEK and report what that covers
#' @export
crypto_shred <- function(kek_id, wrapped_deks) {
  covered <- which(vapply(wrapped_deks,
                          function(w) identical(w$kek_id, kek_id),
                          logical(1))) - 1L
  orphaned <- which(vapply(wrapped_deks,
                           function(w) !identical(w$kek_id, kek_id),
                           logical(1))) - 1L
  list(kek_id = kek_id, records_shredded = length(covered),
       indices = covered,
       still_recoverable = orphaned,
       complete = length(orphaned) == 0L,
       note = paste("any DEK wrapped under a DIFFERENT KEK survives, ",
                    "so a partial rotation leaves data readable"))
}

# house entry point: the package exports one morie_<module>
morie_secrtt <- rotate_kek
