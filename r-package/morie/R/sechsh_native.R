# A hash-chained audit log: tamper-evident, not tamper-proof.
# Sources: Laurie, B., Langley, A. & Kasper, E. (2013) "Certificate
# Transparency", RFC 6962, doi:10.17487/RFC6962 (Merkle Tree Hash,
# leaf/interior domain separation, audit path); Schneier, B. &
# Kelsey, J. (1999) "Secure Audit Logs to Support Computer
# Forensics", ACM TISSEC 2(2), 159-176, doi:10.1145/317087.317089
# (forward integrity through hash chaining and a writer-held key);
# and NIST (2015) Secure Hash Standard (SHS), FIPS PUB 180-4.
#
# Native R arm mirroring the Python arm exactly: the same chain rule
# h_i = H(h_{i-1} || e_i) (or HMAC-SHA-256 with a key), the same
# RFC 6962 Merkle tree hash with 0x00-prefixed leaves and 0x01-
# prefixed interior nodes, and the same top-down recorded audit path
# re-folded bottom-up because the descent and the hashing are
# different directions and combining them the wrong way shows up
# immediately on a non-power-of-two tree.

.LEAF <- as.raw(0x00)
.NODE <- as.raw(0x01)
.GENESIS <- raw(32)

.sechsh_as_bytes <- function(x) {
  if (is.raw(x)) return(x)
  if (is.character(x)) return(charToRaw(paste(x, collapse = "")))
  if (is.null(x)) return(raw(0))
  stop("expected raw, character or NULL")
}

.sechsh_hexlify <- function(bs) {
  paste(format(as.hexmode(as.integer(bs)), width = 2,
               upper.case = TRUE), collapse = "")
}

.constant_time_equal <- function(a, b) {
  if (length(a) != length(b)) return(FALSE)
  v <- as.integer(bitwXor(as.integer(a), as.integer(b)))
  v <- sum(v)
  v == 0L
}

#' One step of the hash chain
#' @export
chain_entry <- function(previous_hash, entry, key = NULL) {
  p <- .sechsh_as_bytes(previous_hash)
  e <- .sechsh_as_bytes(entry)
  if (is.null(key)) {
    return(list(hash = sha256(c(p, e)), keyed = FALSE))
  }
  list(hash = .morie_hmac_sha256_impl(key, c(p, e)), keyed = TRUE,
       note = paste("forward rewriting now needs the KEY as well ",
                    "as write access"))
}

#' Build a complete chain from a list of entries
#' @export
build_chain <- function(entries, key = NULL, genesis = .GENESIS) {
  prev <- .sechsh_as_bytes(genesis)
  hashes <- list()
  for (e in entries) {
    prev <- chain_entry(prev, e, key)$hash
    hashes[[length(hashes) + 1L]] <- prev
  }
  list(hashes = hashes,
       head = if (length(hashes) > 0L) prev else .sechsh_as_bytes(genesis),
       n = length(hashes),
       head_hex = .sechsh_hexlify(if (length(hashes) > 0L) prev
                           else .sechsh_as_bytes(genesis)),
       keyed = !is.null(key))
}

#' Verify a chain, returning the FIRST bad index (not just a boolean)
#' @export
verify_chain <- function(entries, hashes, key = NULL,
                         genesis = .GENESIS) {
  if (length(entries) != length(hashes)) {
    stop("sechsh: ", length(entries), " entries but ",
         length(hashes), " hashes -- an entry or a hash has been ",
         "dropped")
  }
  prev <- .sechsh_as_bytes(genesis)
  first_bad <- NULL
  for (i in seq_along(entries)) {
    want <- chain_entry(prev, entries[[i]], key)$hash
    if (!.constant_time_equal(want, hashes[[i]])) {
      if (is.null(first_bad)) first_bad <- i - 1L
    }
    prev <- .sechsh_as_bytes(hashes[[i]])
  }
  list(estimate = is.null(first_bad), intact = is.null(first_bad),
       first_bad = first_bad,
       verified_through = if (is.null(first_bad))
         length(entries) else first_bad,
       n = length(entries),
       method = "hash-chained audit log; Schneier & Kelsey (1999)",
       note = paste("tamper-EVIDENT, not tamper-proof: an attacker ",
                    "who can rewrite the whole tail recomputes every ",
                    "later hash, which is what keying and external ",
                    "anchoring are for"))
}

#' RFC 6962 Merkle Tree Hash
#' @export
merkle_root <- function(leaves) {
  L <- lapply(leaves, .sechsh_as_bytes)
  if (length(L) == 0L) return(sha256(raw(0)))
  if (length(L) == 1L) return(sha256(c(.LEAF, L[[1L]])))
  k <- 1L
  while (k * 2L < length(L)) k <- k * 2L
  c1 <- do.call(merkle_root, list(L[seq_len(k)]))
  c2 <- do.call(merkle_root, list(L[(k + 1L):length(L)]))
  sha256(c(.NODE, c1, c2))
}

#' Build an inclusion proof (audit path) for the given leaf
#' @export
inclusion_proof <- function(leaves, index) {
  L <- lapply(leaves, .sechsh_as_bytes)
  m <- as.integer(index)
  if (m < 0L || m >= length(L)) {
    stop("sechsh: index ", m, " is outside a log of ",
         length(L))
  }
  path <- list()
  lo <- 0L; hi <- length(L)
  while (hi - lo > 1L) {
    k <- 1L
    while (k * 2L < hi - lo) k <- k * 2L
    if (m - lo < k) {
      path[[length(path) + 1L]] <- merkle_root(L[(lo + k + 1L):hi])
      hi <- lo + k
    } else {
      path[[length(path) + 1L]] <- merkle_root(L[(lo + 1L):(lo + k)])
      lo <- lo + k
    }
  }
  list(path = path,
       path_hex = lapply(path, .sechsh_hexlify),
       length = length(path), index = m, size = length(L),
       note = paste("log2(n) hashes prove membership against a ",
                    "trusted head"))
}

#' Recompute the root from a leaf and an audit path
#' @export
verify_inclusion <- function(leaf, index, size, path, root) {
  m <- as.integer(index); n <- as.integer(size)
  if (m < 0L || m >= n) {
    stop("sechsh: index ", m, " is outside a log of ", n)
  }
  node <- sha256(c(.LEAF, .sechsh_as_bytes(leaf)))
  # The Python collects the descent top-down, then folds bottom-up.
  lo <- 0L; hi <- n
  steps <- list(); used <- 0L
  p <- as.list(path)
  while (hi - lo > 1L) {
    if (used >= length(p)) {
      stop("sechsh: the audit path is too short for a log of ", n)
    }
    k <- 1L
    while (k * 2L < hi - lo) k <- k * 2L
    sib <- .sechsh_as_bytes(p[[used + 1L]])
    used <- used + 1L
    if (m - lo < k) {
      steps[[length(steps) + 1L]] <- list(sib = sib, on_right = TRUE)
      hi <- lo + k
    } else {
      steps[[length(steps) + 1L]] <- list(sib = sib, on_right = FALSE)
      lo <- lo + k
    }
  }
  for (j in rev(seq_along(steps))) {
    st <- steps[[j]]
    if (st$on_right) {
      node <- sha256(c(.NODE, node, st$sib))
    } else {
      node <- sha256(c(.NODE, st$sib, node))
    }
  }
  list(root = node, root_hex = .sechsh_hexlify(node),
       valid = .constant_time_equal(node, root),
       path_used = used)
}

# house entry point: the package exports one morie_<module>
morie_sechsh <- verify_chain
