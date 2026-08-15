# MAFFT: multiple sequence alignment through the fast Fourier transform.
# Sources: Katoh, K., Misawa, K., Kuma, K., & Miyata, T. (2002)
# "MAFFT: a novel method for rapid multiple sequence alignment based
# on fast Fourier transform", Nucleic Acids Research 30(14),
# 3059-3066, doi:10.1093/nar/gkf436 -- the Fourier construction,
# the Grantham volume/polarity vector sequences, the sliding-window
# peak walk, the segment-level DP of Figure 2A, the residue-level
# DP restricted to the sub-matrices between segment centres
# (Figure 2B), the score normalisation of Equation 7, the
# shared-gap penalty of Equation 8, and the three named methods
# FFT-NS-1, FFT-NS-2, FFT-NS-i. Katoh, K., Kuma, K., Toh, H., &
# Miyata, T. (2005) "MAFFT version 5: improvement in accuracy of
# multiple sequence alignment", NAR 33(2), 511-518,
# doi:10.1093/nar/gki198. Katoh, K., & Standley, D. M. (2013) "MAFFT
# Multiple Sequence Alignment Software Version 7", MBE 30(4),
# 772-780, doi:10.1093/molbev/mst010 -- the options table.
#
# Native implementation mirroring Python morie.fn.mafft exactly:
# same Grantham vectors, same correlation by FFT or direct
# summation with circular wrap, same 30/20/0.7/150 sliding-window
# parameters, same score normalisation with the same NW-AP-2
# control whose S_a is 0.8211, same shared-gap penalty, same three
# (plus two NW-NS) methods, same iterative refinement that splits
# along a tree edge and keeps the change only if WSP improves.

.MAFFT_AA <- "ARNDCQEGHILKMFPSTWYV"
.MAFFT_NT <- "ACGT"

GRANTHAM_POLARITY <- list(
  A = 8.1, R = 10.5, N = 11.6, D = 13.0, C = 5.5,
  Q = 10.5, E = 12.3, G = 9.0, H = 10.4, I = 5.2,
  L = 4.9, K = 11.3, M = 5.7, F = 5.2, P = 8.0,
  S = 9.2, T = 8.6, W = 5.4, Y = 6.2, V = 5.9)

GRANTHAM_VOLUME <- list(
  A = 31.0, R = 124.0, N = 56.0, D = 54.0, C = 55.0,
  Q = 85.0, E = 83.0, G = 3.0, H = 96.0, I = 111.0,
  L = 111.0, K = 119.0, M = 105.0, F = 132.0, P = 32.5,
  S = 32.0, T = 61.0, W = 170.0, Y = 136.0, V = 84.0)

.MAFFT_METHODS <- c("FFT-NS-1", "FFT-NS-2", "FFT-NS-i",
                    "NW-NS-1", "NW-NS-2")
.MAFFT_MATRICES <- c("normalized", "all_positive")
.MAFFT_SIX_GROUPS <- c("AGPST", "C", "DENQ", "FWY", "HKR", "ILMV")

.mafft_norm <- function(vals) {
  n <- length(vals)
  mu <- sum(vals) / n
  sd <- sqrt(sum((vals - mu)^2) / n)
  if (sd <= 0) stop("mafft: a property with no variation cannot be normalised")
  c(mu, sd)
}

# .MAFFT_AA is one string; the Python arm iterates its characters
# ("for a in _AA"). Indexing the property list by the whole string
# matches no name, so every one of these was NULL and the package
# failed to load: .mafft_norm(NULL) has n = 0 and sd = NaN.
.MAFFT_AA_CHARS <- strsplit(.MAFFT_AA, "")[[1L]]

.MAFFT_P <- .mafft_norm(unlist(GRANTHAM_POLARITY[.MAFFT_AA_CHARS]))
.MAFFT_V <- .mafft_norm(unlist(GRANTHAM_VOLUME[.MAFFT_AA_CHARS]))
.MAFFT_PMU <- .MAFFT_P[1]; .MAFFT_PSD <- .MAFFT_P[2]
.MAFFT_VMU <- .MAFFT_V[1]; .MAFFT_VSD <- .MAFFT_V[2]

.MAFFT_VHAT <- setNames(
  (unlist(GRANTHAM_VOLUME[.MAFFT_AA_CHARS]) - .MAFFT_VMU) / .MAFFT_VSD,
  .MAFFT_AA_CHARS)
.MAFFT_PHAT <- setNames(
  (unlist(GRANTHAM_POLARITY[.MAFFT_AA_CHARS]) - .MAFFT_PMU) / .MAFFT_PSD,
  .MAFFT_AA_CHARS)

mafft_clean <- function(seqs, seq_type = NULL) {
  out <- as.character(toupper(unlist(seqs)))
  if (any(nchar(out) == 0L))
    stop("mafft: an empty sequence was given")
  if (is.null(seq_type)) {
    letters <- unique(unlist(strsplit(paste(out, collapse = ""), "")))
    letters <- setdiff(letters, c("-", "."))
    seq_type <- if (length(letters) > 0 &&
                    all(letters %in% c(.MAFFT_NT, "U", "N"))) "nt" else "aa"
  }
  if (!(seq_type %in% c("aa", "nt")))
    stop("mafft: seq_type must be 'aa' or 'nt'")
  list(seqs = out, seq_type = seq_type)
}

residue_vectors <- function(group, weights = NULL, seq_type = "aa") {
  rows <- as.character(toupper(unlist(group)))
  if (length(rows) == 0L)
    stop("mafft: an empty group has no vectors")
  L <- nchar(rows[1])
  if (any(nchar(rows) != L))
    stop("mafft: sequences in a group must be aligned to the same length")
  if (is.null(weights)) weights <- rep(1.0 / length(rows), length(rows))
  weights <- as.numeric(weights)
  if (length(weights) != length(rows))
    stop("mafft: one weight per sequence is required")
  split_rows <- strsplit(rows, "")
  if (seq_type == "nt") {
    comps <- list()
    for (base in strsplit(.MAFFT_NT, "")[[1]]) {
      comps[[length(comps) + 1L]] <- vapply(split_rows, function(r) {
        sum(weights[r == base])
      }, numeric(L))
    }
    return(comps)
  }
  vol <- vapply(seq_len(L), function(n) {
    sum(vapply(seq_along(split_rows), function(i) {
      ch <- split_rows[[i]][n]
      weights[i] * if (ch %in% .MAFFT_AA_CHARS) .MAFFT_VHAT[[ch]] else 0
    }, numeric(1)))
  }, numeric(1))
  pol <- vapply(seq_len(L), function(n) {
    sum(vapply(seq_along(split_rows), function(i) {
      ch <- split_rows[[i]][n]
      weights[i] * if (ch %in% .MAFFT_AA_CHARS) .MAFFT_PHAT[[ch]] else 0
    }, numeric(1)))
  }, numeric(1))
  list(vol, pol)
}

mafft_xcorr_fft <- function(a, b) {
  n <- length(a); m <- length(b)
  size <- 1L
  while (size < n + m) size <- size * 2L
  fa <- fft(c(as.numeric(a), rep(0, size - n)))
  fb <- fft(c(as.numeric(b), rep(0, size - m)))
  back <- fft(fa * Conj(fb), inverse = TRUE) / size
  list(c = Re(back), size = size)
}

mafft_xcorr_direct <- function(a, b, size) {
  out <- rep(0.0, size)
  n <- length(a); m <- length(b)
  for (k in 0:(size - 1L)) {
    tot <- 0.0
    for (i in 0:(n - 1L)) {
      j <- (i + k) %% size
      if (j < m) tot <- tot + a[i + 1L] * b[j + 1L]
    }
    out[k + 1L] <- tot
  }
  out
}

correlation <- function(group1, group2, weights1 = NULL, weights2 = NULL,
                        seq_type = "aa", method = "fft") {
  if (!(method %in% c("fft", "direct")))
    stop("mafft: method must be 'fft' or 'direct'")
  c1 <- residue_vectors(group1, weights1, seq_type)
  c2 <- residue_vectors(group2, weights2, seq_type)
  total <- NULL; size <- NULL
  for (i in seq_along(c1)) {
    a <- c1[[i]]; b <- c2[[i]]
    if (method == "fft") {
      r <- mafft_xcorr_fft(a, b); part <- r$c; size <- r$size
    } else {
      if (is.null(size)) {
        size <- 1L
        while (size < length(a) + length(b)) size <- size * 2L
      }
      part <- mafft_xcorr_direct(a, b, size)
    }
    if (is.null(total)) total <- part else total <- total + part
  }
  half <- size %/% 2L
  lags <- c(seq_len(half) - 1L, -seq(half, 1L, by = -1L))
  list(lags = lags, c = total, size = size)
}

mafft_peaks <- function(lags, c, n_peaks) {
  ord <- order(c, decreasing = TRUE)
  lags[ord[seq_len(as.integer(n_peaks))]]
}

.MAFFT_JTT_FREQ <- c(0.077, 0.051, 0.043, 0.052, 0.020, 0.041, 0.062,
                     0.074, 0.023, 0.052, 0.091, 0.059, 0.024, 0.040,
                     0.051, 0.069, 0.059, 0.014, 0.032, 0.066)
.MAFFT_JTT_COUNTS <- c(247,216,116,386,48,1433,106,125,32,13,208,750,
                       159,130,9,600,119,180,2914,8,1027,1183,614,291,
                       577,98,84,610,46,446,466,144,40,635,41,41,173,
                       76,130,37,19,20,43,25,26,257,205,63,34,36,314,
                       65,56,134,1324,200,2348,758,102,7,858,754,142,
                       85,75,94,100,61,39,27,23,52,30,27,21,704,974,
                       103,51,16,15,8,66,9,13,18,50,196,1093,7,49,
                       901,217,31,39,15,395,71,93,157,31,578,77,23,36,
                       2413,413,1738,244,353,182,156,1131,138,172,436,
                       228,54,309,1138,2440,230,693,151,66,149,142,164,
                       76,930,172,398,343,39,412,2258,11,109,2,5,38,
                       12,12,69,5,12,82,9,8,37,6,36,8,41,46,114,89,
                       164,40,15,15,514,61,84,20,17,850,22,164,45,41,
                       1766,69,55,127,99,58,226,276,22,3938,1261,58,
                       559,189,84,219,526,27,42)

mafft_jtt_exchangeability <- function() {
  f <- .MAFFT_JTT_FREQ
  S <- matrix(0, 20, 20)
  k <- 1L
  for (i in 2:20) {
    for (j in 1:(i - 1L)) {
      n <- .MAFFT_JTT_COUNTS[k]
      k <- k + 1L
      v <- n / (400.0 * f[i] * f[j])
      S[i, j] <- v
      S[j, i] <- v
    }
  }
  list(S = S, f = f)
}

jtt_matrix <- function(pam = 200, scale = 10.0) {
  if (pam <= 0) stop("mafft: pam must be positive")
  ex <- mafft_jtt_exchangeability()
  S <- ex$S; f <- ex$f
  Q <- matrix(0, 20, 20)
  for (i in 1:20) {
    off <- 0.0
    for (j in 1:20) {
      if (i == j) next
      Q[i, j] <- S[i, j] * f[j]
      off <- off + Q[i, j]
    }
    Q[i, i] <- -off
  }
  mu <- -sum(f * diag(Q))
  Q <- Q / (mu * 100.0)
  rt <- sqrt(f)
  A <- Q * (rt / rep(rt, each = 20))
  es <- eigen(A, symmetric = TRUE)
  w <- es$values; V <- es$vectors
  e <- exp(w * as.numeric(pam))
  P <- matrix(0, 20, 20)
  for (i in 1:20) {
    for (j in 1:20) {
      P[i, j] <- sum(V[i, ] * e * V[j, ]) * rt[j] / rt[i]
    }
  }
  M <- list()
  for (i in 1:20) for (j in 1:20) {
    a <- substr(.MAFFT_AA, i, i); b <- substr(.MAFFT_AA, j, j)
    p <- max(P[i, j], 1e-300)
    M[[paste0(a, "_", b)]] <- scale * log10(p / f[j])
  }
  M_named <- M
  attr(M_named, "keys") <- outer(strsplit(.MAFFT_AA, "")[[1]],
                                strsplit(.MAFFT_AA, "")[[1]],
                                function(a, b) paste0(a, "_", b))
  list(matrix = M_named, freqs = setNames(as.list(f), strsplit(.MAFFT_AA, "")[[1]]),
       P = P, Q = Q, pam = pam,
       rate = -sum(f * diag(Q)))
}

mafft_default_raw <- function(seq_type, which = "jtt200") {
  if (seq_type == "nt") {
    M <- list()
    for (a in strsplit(.MAFFT_NT, "")[[1]])
      for (b in strsplit(.MAFFT_NT, "")[[1]])
        M[[paste0(a, "_", b)]] <- if (a == b) 1.0 else -1.0
    return(list(M = M, f = NULL))
  }
  if (which == "grantham") {
    M <- list()
    aa_letters <- strsplit(.MAFFT_AA, "")[[1]]
    for (a in aa_letters) for (b in aa_letters) {
      M[[paste0(a, "_", b)]] <- -((.MAFFT_VHAT[[a]] - .MAFFT_VHAT[[b]])^2 +
                                   (.MAFFT_PHAT[[a]] - .MAFFT_PHAT[[b]])^2)
    }
    return(list(M = M, f = NULL))
  }
  j <- jtt_matrix(200)
  list(M = j$matrix, f = j$freqs)
}

mafft_lookup <- function(M, a, b) {
  v <- M[[paste0(a, "_", b)]]
  if (is.null(v)) 0.0 else as.numeric(v)
}

normalized_similarity_matrix <- function(raw_matrix = NULL, freqs = NULL,
                                         s_a = 0.06, seq_type = "aa",
                                         mode = "normalized",
                                         default = "jtt200") {
  if (!(mode %in% .MAFFT_MATRICES))
    stop(sprintf("mafft: mode must be one of %s",
                 paste(.MAFFT_MATRICES, collapse = ", ")))
  if (!(default %in% c("jtt200", "grantham")))
    stop("mafft: default must be 'jtt200' or 'grantham'")
  alpha <- if (seq_type == "nt") strsplit(.MAFFT_NT, "")[[1]] else strsplit(.MAFFT_AA, "")[[1]]
  default_f <- NULL
  if (is.null(raw_matrix)) {
    d <- mafft_default_raw(seq_type, default)
    M <- d$M; default_f <- d$f
  } else {
    M <- raw_matrix
  }
  if (is.null(freqs) && !is.null(default_f)) freqs <- default_f
  if (is.null(freqs))
    freqs <- setNames(as.list(rep(1.0 / length(alpha), length(alpha))), alpha)
  f_vec <- vapply(alpha, function(a) as.numeric(freqs[[a]] %||% 0), numeric(1))
  tot <- sum(f_vec)
  if (tot <= 0) stop("mafft: frequencies must be positive")
  f_vec <- f_vec / tot
  names(f_vec) <- alpha
  for (a in alpha) for (b in alpha) {
    if (is.null(M[[paste0(a, "_", b)]]))
      stop(sprintf("mafft: raw_matrix is missing (%s, %s)", a, b))
  }
  avg1 <- sum(f_vec * vapply(alpha, function(a) mafft_lookup(M, a, a), numeric(1)))
  avg2 <- sum(outer(f_vec, f_vec) *
              outer(alpha, alpha, Vectorize(function(a, b) mafft_lookup(M, a, b))))
  if (abs(avg1 - avg2) < 1e-15)
    stop("mafft: raw_matrix has no signal (average1 equals average2)")
  base <- list()
  for (a in alpha) for (b in alpha)
    base[[paste0(a, "_", b)]] <- (mafft_lookup(M, a, b) - avg2) / (avg1 - avg2)
  if (mode == "all_positive")
    s_a <- -min(unlist(base))
  out <- list()
  for (k in names(base)) out[[k]] <- base[[k]] + s_a
  list(matrix = out, s_a = as.numeric(s_a), alphabet = alpha,
       average1 = avg1, average2 = avg2, freqs = as.list(f_vec), mode = mode)
}

`%||%` <- function(x, y) if (is.null(x)) y else x

mafft_site_score <- function(M, ga, gb, wa, wb, i, j) {
  tot <- 0.0
  for (idx_a in seq_along(ga)) {
    a <- substr(ga[idx_a], i, i)
    if (a == "-") next
    for (idx_b in seq_along(gb)) {
      b <- substr(gb[idx_b], j, j)
      if (b == "-") next
      tot <- tot + wa[idx_a] * wb[idx_b] * mafft_lookup(M, a, b)
    }
  }
  tot
}

mafft_gap_profiles <- function(group, weights) {
  L <- nchar(group[1])
  gs <- rep(0.0, L + 1L); ge <- rep(0.0, L + 1L)
  for (idx in seq_along(group)) {
    s <- group[idx]
    w <- weights[idx]
    z <- vapply(strsplit(s, "")[[1]], function(ch) if (ch == "-") 1.0 else 0.0,
                numeric(1))
    a <- 1.0 - z
    for (x in 1:L) {
      nxt <- if (x < L) z[x + 1L] else 0.0
      gs[x] <- gs[x] + w * a[x] * nxt
      prv <- if (x > 1L) z[x - 1L] else 0.0
      ge[x] <- ge[x] + w * prv * a[x]
    }
  }
  list(gs = gs, ge = ge)
}

mafft_nw <- function(g1, g2, M, w1, w2, s_op) {
  n <- nchar(g1[1]); m <- nchar(g2[1])
  if (n == 0) {
    return(list(out1 = rep(strrep("-", m), length(g1)),
                out2 = as.list(g2)))
  }
  if (m == 0) {
    return(list(out1 = as.list(g1),
                out2 = rep(strrep("-", n), length(g2))))
  }
  gp1 <- mafft_gap_profiles(g1, w1)
  gp2 <- mafft_gap_profiles(g2, w2)
  gs1 <- gp1$gs; ge1 <- gp1$ge
  gs2 <- gp2$gs; ge2 <- gp2$ge
  neg <- -Inf
  P <- matrix(neg, n + 1L, m + 1L)
  back <- vector("list", (n + 1L) * (m + 1L))
  dim(back) <- c(n + 1L, m + 1L)
  P[1, 1] <- 0.0
  for (i in 2:(n + 1L)) {
    P[i, 1] <- -s_op * (1.0 - (gs1[1] + ge1[i - 1L]) / 2.0)
    back[[i, 1]] <- list(kind = "I", pi = 0L, pj = 0L)
  }
  for (j in 2:(m + 1L)) {
    P[1, j] <- -s_op * (1.0 - (gs2[1] + ge2[j - 1L]) / 2.0)
    back[[1, j]] <- list(kind = "D", pi = 0L, pj = 0L)
  }
  for (i in 2:(n + 1L)) {
    for (j in 2:(m + 1L)) {
      h <- mafft_site_score(M, g1, g2, w1, w2, i - 1L, j - 1L)
      best <- list(v = P[i - 1L, j - 1L], kind = "M", pi = i - 1L, pj = j - 1L)
      for (x in 0:(i - 1L)) {
        if (is.infinite(P[x + 1L, j - 1L + 1L])) next
        pen <- s_op * (1.0 - (gs1[x + 1L] + ge1[i]) / 2.0)
        v <- P[x + 1L, j] - pen
        if (v > best$v)
          best <- list(v = v, kind = "I", pi = x, pj = j - 1L)
      }
      for (y in 0:(j - 1L)) {
        if (is.infinite(P[i, y + 1L])) next
        pen <- s_op * (1.0 - (gs2[y + 1L] + ge2[j]) / 2.0)
        v <- P[i, y + 1L] - pen
        if (v > best$v)
          best <- list(v = v, kind = "D", pi = i - 1L, pj = y)
      }
      P[i, j] <- h + best$v
      back[[i, j]] <- list(kind = best$kind, pi = best$pi, pj = best$pj)
    }
  }
  cols <- list()
  i <- n + 1L; j <- m + 1L
  while (i > 1L && j > 1L) {
    b <- back[[i, j]]
    cols[[length(cols) + 1L]] <- c(i - 1L, j - 1L)
    if (b$kind == "I") {
      for (t in (i - 2L):(b$pi)) cols[[length(cols) + 1L]] <- c(t, NA)
    } else if (b$kind == "D") {
      for (t in (j - 2L):(b$pj)) cols[[length(cols) + 1L]] <- c(NA, t)
    }
    i <- b$pi + 1L; j <- b$pj + 1L
  }
  for (t in (i - 1L):1) cols[[length(cols) + 1L]] <- c(t, NA)
  for (t in (j - 1L):1) cols[[length(cols) + 1L]] <- c(NA, t)
  cols <- rev(cols)
  out1 <- vapply(g1, function(s) {
    paste(vapply(cols, function(c) {
      if (is.na(c[1])) "-" else substr(s, c[1], c[1])
    }, character(1)), collapse = "")
  }, character(1))
  out2 <- vapply(g2, function(s) {
    paste(vapply(cols, function(c) {
      if (is.na(c[2])) "-" else substr(s, c[2], c[2])
    }, character(1)), collapse = "")
  }, character(1))
  list(out1 = as.list(out1), out2 = as.list(out2))
}

group_align <- function(group1, group2, scoring, weights1 = NULL,
                        weights2 = NULL, s_op = 2.4, anchors = NULL) {
  g1 <- as.character(toupper(unlist(group1)))
  g2 <- as.character(toupper(unlist(group2)))
  if (length(g1) == 0L || length(g2) == 0L)
    stop("mafft: both groups must be non-empty")
  for (g in list(g1, g2))
    if (length(unique(nchar(g))) != 1L)
      stop("mafft: a group must be aligned to one length")
  w1 <- if (is.null(weights1)) rep(1.0 / length(g1), length(g1)) else as.numeric(weights1)
  w2 <- if (is.null(weights2)) rep(1.0 / length(g2), length(g2)) else as.numeric(weights2)
  if (length(w1) != length(g1) || length(w2) != length(g2))
    stop("mafft: one weight per sequence is required")
  M <- if (is.list(scoring)) scoring$matrix else scoring
  if (!is.null(anchors)) {
    n <- nchar(g1[1]); m <- nchar(g2[1])
    given <- unique(do.call(rbind, lapply(anchors, function(a) a[1:2])))
    given <- given[order(given[, 1]), , drop = FALSE]
    for (k in 2:nrow(given))
      if (given[k, 2] < given[k - 1L, 2])
        stop("mafft: anchors cross and cannot lie on one alignment path")
    for (r in seq_len(nrow(given))) {
      if (given[r, 1] < 0 || given[r, 1] > n || given[r, 2] < 0 || given[r, 2] > m)
        stop("mafft: an anchor is outside the groups")
    }
    pts <- rbind(c(0, 0), given, c(n, m))
    out1 <- rep("", length(g1)); out2 <- rep("", length(g2))
    prev <- pts[1, , drop = FALSE]
    for (idx in 2:nrow(pts)) {
      pt <- pts[idx, , drop = FALSE]
      a1 <- substr(g1, prev[1, 1] + 1L, pt[1])
      a2 <- substr(g2, prev[1, 2] + 1L, pt[2])
      if (pt[1] == prev[1, 1] && pt[2] == prev[1, 2]) { prev <- pt; next }
      r <- mafft_nw(a1, a2, M, w1, w2, s_op)
      out1 <- paste0(out1, unlist(r$out1))
      out2 <- paste0(out2, unlist(r$out2))
      prev <- pt
    }
    return(list(out1 = as.list(out1), out2 = as.list(out2)))
  }
  mafft_nw(g1, g2, M, w1, w2, s_op)
}

find_homologous_segments <- function(group1, group2, scoring,
                                     weights1 = NULL, weights2 = NULL,
                                     seq_type = "aa", window = 30L,
                                     n_peaks = 20L, threshold = 0.7,
                                     max_len = 150L, corr_method = "fft") {
  if (window < 1L || n_peaks < 1L || max_len < 1L)
    stop("mafft: window, n_peaks and max_len must be positive")
  M <- if (is.list(scoring)) scoring$matrix else scoring
  g1 <- as.character(toupper(unlist(group1)))
  g2 <- as.character(toupper(unlist(group2)))
  w1 <- if (is.null(weights1)) rep(1.0 / length(g1), length(g1)) else as.numeric(weights1)
  w2 <- if (is.null(weights2)) rep(1.0 / length(g2), length(g2)) else as.numeric(weights2)
  n <- nchar(g1[1]); m <- nchar(g2[1])
  cr <- correlation(g1, g2, w1, w2, seq_type, corr_method)
  segs <- list()
  for (k in mafft_peaks(cr$lags, cr$c, n_peaks)) {
    lo <- max(0, -k); hi <- min(n, m - k)
    if (hi - lo < window) next
    run <- NULL
    for (start in lo:(hi - window)) {
      score <- sum(vapply(0:(window - 1L), function(t) {
        mafft_site_score(M, g1, g2, w1, w2, start + t + 1L,
                         start + t + k + 1L)
      }, numeric(1))) / window
      if (score > threshold) {
        if (is.null(run))
          run <- list(start = start, end = start + window, scores = c(score))
        else {
          run$end <- start + window
          run$scores <- c(run$scores, score)
        }
      } else if (!is.null(run)) {
        segs[[length(segs) + 1L]] <- c(run$start, run$start + k,
                                        run$end - run$start,
                                        sum(run$scores) / length(run$scores),
                                        k)
        run <- NULL
      }
    }
    if (!is.null(run))
      segs[[length(segs) + 1L]] <- c(run$start, run$start + k,
                                      run$end - run$start,
                                      sum(run$scores) / length(run$scores),
                                      k)
  }
  cut <- list()
  for (s in segs) {
    s1 <- s[1]; s2 <- s[2]; ln <- s[3]; sc <- s[4]; k <- s[5]
    while (ln > max_len) {
      cut[[length(cut) + 1L]] <- c(s1, s2, max_len, sc, k)
      s1 <- s1 + max_len; s2 <- s2 + max_len; ln <- ln - max_len
    }
    if (ln > 0) cut[[length(cut) + 1L]] <- c(s1, s2, ln, sc, k)
  }
  if (length(cut) == 0L) return(matrix(numeric(0), 0, 5))
  out <- do.call(rbind, cut)
  out <- out[order(out[, 1]), , drop = FALSE]
  out
}

arrange_segments <- function(segments) {
  if (is.matrix(segments)) segs <- as.list(seq_len(nrow(segments))) else
    segs <- as.list(seq_along(segments))
  if (length(segs) == 0L) return(list())
  if (is.matrix(segments)) {
    idx <- order(segments[, 1])
    segments <- segments[idx, , drop = FALSE]
  }
  n <- nrow(segments)
  if (n == 0L) return(matrix(numeric(0), 0, 5))
  best <- segments[, 3] * segments[, 4]
  prev <- rep(NA, n)
  for (i in 1:n) {
    for (j in 1:(i - 1L)) {
      a <- segments[j, ]; b <- segments[i, ]
      if (a[1] + a[3] <= b[1] && a[2] + a[3] <= b[2]) {
        v <- best[j] + segments[i, 3] * segments[i, 4]
        if (v > best[i]) { best[i] <- v; prev[i] <- j }
      }
    }
  }
  end <- which.max(best)
  chain <- list()
  while (!is.na(end)) {
    chain[[length(chain) + 1L]] <- segments[end, ]
    end <- prev[end]
  }
  do.call(rbind, rev(chain))
}

mafft_anchors_from <- function(chain) {
  if (!is.matrix(chain) || nrow(chain) == 0L) return(list())
  mapply(seq_len(nrow(chain)), seq_len(nrow(chain)), FUN = function(i, j) {
    s <- chain[i, ]
    if (s[1] >= 0 && s[2] >= 0)
      c(s[1] + (s[3] %/% 2), s[2] + (s[3] %/% 2)) else NULL
  }, SIMPLIFY = FALSE)
}

sixtuple_distance <- function(seqs) {
  coded <- vapply(seqs, function(s) {
    t <- ""
    s_up <- toupper(as.character(s))
    for (ch in strsplit(s_up, "")[[1]]) {
      if (ch == "-") next
      matched <- FALSE
      for (gi in seq_along(.MAFFT_SIX_GROUPS)) {
        if (ch %in% strsplit(.MAFFT_SIX_GROUPS[gi], "")[[1]]) {
          t <- paste0(t, rawToChar(as.raw(96L + gi)))
          matched <- TRUE; break
        }
      }
      if (!matched) t <- paste0(t, "z")
    }
    t
  }, character(1))
  tuples <- function(t) {
    d <- list()
    if (nchar(t) < 6L) return(d)
    for (i in 1:(nchar(t) - 5L)) {
      key <- substr(t, i, i + 5L)
      d[[key]] <- if (is.null(d[[key]])) 1L else d[[key]] + 1L
    }
    d
  }
  tabs <- lapply(coded, tuples)
  shared <- function(a, b) {
    s <- 0L
    for (k in names(a)) s <- s + min(a[[k]], b[[k]] %||% 0L)
    s
  }
  n <- length(seqs)
  D <- matrix(0, n, n)
  for (i in 1:n) for (j in 1:n) {
    if (i == j) next
    denom <- min(shared(tabs[[i]], tabs[[i]]), shared(tabs[[j]], tabs[[j]]))
    t_val <- shared(tabs[[i]], tabs[[j]])
    D[i, j] <- 1.0 - if (denom) t_val / denom else 0
  }
  D
}

guide_tree <- function(D) {
  n <- nrow(D)
  if (n < 2L) stop("mafft: a guide tree needs at least two sequences")
  clusters <- list()
  for (i in 1:n) clusters[[i]] <- i
  dist <- list()
  for (i in 1:n) for (j in 1:n) if (i != j) dist[[paste(i, j)]] <- D[i, j]
  merges <- list()
  nxt <- n
  active <- as.list(1:n)
  while (length(active) > 1L) {
    best <- NULL
    for (k in seq_along(active)) {
      for (kk in (k + 1L):length(active)) {
        i <- active[[k]]; j <- active[[kk]]
        d <- dist[[paste(i, j)]]
        key <- c(d, i, j)
        if (is.null(best) || d < best[1] ||
            (d == best[1] && (i < best[2] || (i == best[2] && j < best[3]))))
          best <- key
      }
    }
    i <- best[2]; j <- best[3]
    members <- c(clusters[[i]], clusters[[j]])
    merges[[length(merges) + 1L]] <- list(i = i, j = j, new = nxt,
                                          members = members)
    for (k in seq_along(active)) {
      idx <- active[[k]]
      if (idx %in% c(i, j)) next
      ni <- length(clusters[[i]]); nj <- length(clusters[[j]])
      d <- (ni * dist[[paste(i, idx)]] + nj * dist[[paste(j, idx)]]) /
           (ni + nj)
      dist[[paste(nxt, idx)]] <- d
      dist[[paste(idx, nxt)]] <- d
    }
    clusters[[nxt + 1L]] <- members
    active <- c(active[!(active %in% c(i, j))], list(nxt))
    nxt <- nxt + 1L
  }
  merges
}

mafft_weights <- function(k) rep(1.0 / k, k)

progressive_align <- function(seqs, scoring, tree = NULL, seq_type = "aa",
                              s_op = 2.4, use_fft = TRUE, ...) {
  seqs <- as.character(toupper(unlist(seqs)))
  if (length(seqs) < 2L)
    stop("mafft: at least two sequences are needed")
  if (is.null(tree)) tree <- guide_tree(sixtuple_distance(seqs))
  profiles <- list(); members <- list()
  for (i in seq_along(seqs)) {
    profiles[[as.character(i)]] <- seqs[i]
    members[[as.character(i)]] <- i
  }
  for (m in tree) {
    i <- m$i; j <- m$j; new <- m$new
    g1 <- profiles[[as.character(i)]]
    g2 <- profiles[[as.character(j)]]
    if (!is.list(g1)) g1 <- list(g1)
    if (!is.list(g2)) g2 <- list(g2)
    anchors <- NULL
    if (use_fft) {
      segs <- find_homologous_segments(g1, g2, scoring,
                                       mafft_weights(length(g1)),
                                       mafft_weights(length(g2)),
                                       seq_type, ...)
      arr <- arrange_segments(segs)
      if (is.matrix(arr) && nrow(arr) > 0L)
        anchors <- mafft_anchors_from(arr)
    }
    a <- group_align(g1, g2, scoring,
                     mafft_weights(length(g1)),
                     mafft_weights(length(g2)), s_op, anchors)
    profiles[[as.character(new)]] <- c(a$out1, a$out2)
    members[[as.character(new)]] <- c(members[[as.character(i)]],
                                       members[[as.character(j)]])
    profiles[[as.character(i)]] <- NULL
    profiles[[as.character(j)]] <- NULL
  }
  root <- names(profiles)[1]
  ord <- members[[root]]
  out <- vector("list", length(seqs))
  for (pos in seq_along(ord)) out[[ord[pos]]] <- profiles[[root]][pos]
  out
}

wsp_score <- function(alignment, scoring, s_op = 2.4, weights = NULL) {
  aln <- as.character(toupper(unlist(alignment)))
  if (length(unique(nchar(aln))) != 1L)
    stop("mafft: an alignment must be rectangular")
  M <- if (is.list(scoring)) scoring$matrix else scoring
  k <- length(aln)
  w <- if (is.null(weights)) mafft_weights(k) else as.numeric(weights)
  total <- 0.0
  for (i in 1:(k - 1L))
    for (j in (i + 1L):k) {
      pair <- w[i] * w[j]
      opened <- FALSE
      ai <- strsplit(aln[i], "")[[1]]; bj <- strsplit(aln[j], "")[[1]]
      for (idx in seq_along(ai)) {
        a <- ai[idx]; b <- bj[idx]
        if (a == "-" || b == "-") {
          if (!opened) { total <- total - pair * s_op; opened <- TRUE }
        } else {
          opened <- FALSE
          total <- total + pair * mafft_lookup(M, a, b)
        }
      }
    }
  total
}

mafft_degap <- function(group) {
  if (length(group) == 0L) return(group)
  L <- nchar(group[1])
  keep <- c()
  for (i in 1:L) {
    col <- substr(group, i, i)
    if (any(col != "-")) keep <- c(keep, i)
  }
  vapply(group, function(s) {
    if (length(keep) == 0L) "" else paste(substr(rep(s, length(keep)),
                                                keep, keep), collapse = "")
  }, character(1))
}

iterative_refine <- function(alignment, scoring, tree = NULL, s_op = 2.4,
                             max_iterate = 16L, seq_type = "aa",
                             use_fft = TRUE, ...) {
  aln <- as.character(toupper(unlist(alignment)))
  max_iterate <- as.integer(max_iterate)
  if (max_iterate < 1L) stop("mafft: max_iterate must be at least 1")
  best <- wsp_score(aln, scoring, s_op)
  if (is.null(tree))
    tree <- guide_tree(sixtuple_distance(gsub("-", "", aln, fixed = TRUE)))
  groups <- list()
  for (idx in seq_along(tree)) {
    m <- tree[[idx]]
    rest <- setdiff(seq_along(aln), m$members)
    if (length(m$members) > 0L && length(rest) > 0L)
      groups[[length(groups) + 1L]] <- list(members = m$members,
                                            rest = rest)
  }
  rounds <- 0L
  for (it in seq_len(max_iterate)) {
    improved <- FALSE
    for (g in groups) {
      members <- g$members; rest <- g$rest
      g1 <- mafft_degap(aln[members])
      g2 <- mafft_degap(aln[rest])
      anchors <- NULL
      if (use_fft) {
        segs <- find_homologous_segments(g1, g2, scoring,
                                         mafft_weights(length(g1)),
                                         mafft_weights(length(g2)),
                                         seq_type, ...)
        arr <- arrange_segments(segs)
        if (is.matrix(arr) && nrow(arr) > 0L)
          anchors <- mafft_anchors_from(arr)
      }
      a <- group_align(g1, g2, scoring,
                       mafft_weights(length(g1)),
                       mafft_weights(length(g2)), s_op, anchors)
      cand <- vector("list", length(aln))
      for (pos in seq_along(members)) cand[[members[pos]]] <- a$out1[[pos]]
      for (pos in seq_along(rest)) cand[[rest[pos]]] <- a$out2[[pos]]
      cand <- unlist(cand)
      sc <- wsp_score(cand, scoring, s_op)
      if (sc > best + 1e-12) { aln <- cand; best <- sc; improved <- TRUE }
    }
    rounds <- rounds + 1L
    if (!improved) break
  }
  list(aln = aln, score = best, rounds = rounds)
}

mafft_alignment <- function(sequences, method = "FFT-NS-2",
                            seq_type = NULL, raw_matrix = NULL,
                            freqs = NULL, s_a = 0.06, s_op = 2.4,
                            matrix = "normalized", window = 30L,
                            n_peaks = 20L, threshold = 0.7,
                            max_len = 150L, max_iterate = 16L) {
  if (!(method %in% .MAFFT_METHODS))
    stop(sprintf("mafft: method must be one of %s",
                 paste(.MAFFT_METHODS, collapse = ", ")))
  cl <- mafft_clean(sequences, seq_type)
  seqs <- cl$seqs; kind <- cl$seq_type
  if (length(seqs) < 2L) stop("mafft: at least two sequences are needed")
  sc <- normalized_similarity_matrix(raw_matrix, freqs, s_a, kind, matrix)
  use_fft <- startsWith(method, "FFT")
  kw <- list(window = window, n_peaks = n_peaks, threshold = threshold,
             max_len = max_len)

  tree1 <- guide_tree(sixtuple_distance(seqs))
  aln <- progressive_align(seqs, sc, tree1, kind, s_op, use_fft, kw$window,
                           kw$n_peaks, kw$threshold, kw$max_len)
  tree_used <- tree1; rounds <- 0L
  if (method %in% c("FFT-NS-2", "NW-NS-2", "FFT-NS-i")) {
    tree2 <- guide_tree(sixtuple_distance(aln))
    aln <- progressive_align(seqs, sc, tree2, kind, s_op, use_fft,
                             kw$window, kw$n_peaks, kw$threshold, kw$max_len)
    tree_used <- tree2
  }
  score <- wsp_score(aln, sc, s_op)
  if (method == "FFT-NS-i") {
    rr <- iterative_refine(aln, sc, tree_used, s_op, max_iterate, kind,
                           use_fft, kw$window, kw$n_peaks, kw$threshold,
                           kw$max_len)
    aln <- rr$aln; score <- rr$score; rounds <- rr$rounds
  }
  list(estimate = aln, alignment = aln, score = as.numeric(score),
       method = method, seq_type = kind, length = nchar(aln[[1]]),
       n = length(seqs), s_a = sc$s_a, s_op = as.numeric(s_op),
       matrix_mode = matrix, tree = tree_used, refine_rounds = rounds,
       note = paste("Katoh et al. 2002: the FFT finds homologous segments",
                    "and the residue DP is restricted to the sub-matrices",
                    "between their centres; NW-NS-* skip the FFT and",
                    "matrix='all_positive' is the paper's NW-AP-2 control,",
                    "whose S_a comes out at 0.8211 against the 0.82 the",
                    "paper prints. The default raw matrix is the paper's",
                    "own 200-PAM JTT log-odds; default='grantham' builds",
                    "one from the volume/polarity vectors instead."))
}

mafftalignment <- mafft_alignment

.mafft_cheatsheet <- function() {
  paste("mafft: MAFFT (Katoh et al. 2002). Residues become Grantham ",
        "volume/polarity vectors, c(k) = c_v(k) + c_p(k) is got by ",
        "FFT as V1*(m).V2(m), a 30-site window over the top 20 peaks ",
        "at 0.7/site gives homologous segments (merged, then cut at ",
        "150), a segment DP arranges them, and the residue DP runs ",
        "only between their centres. Equation 7 rescales any matrix ",
        "so random sequence scores S_a and identity scores 1 + S_a; ",
        "the gap penalty S_op{1 - [g_start + g_end]/2} is zero where ",
        "the group already has that gap. method= FFT-NS-1, FFT-NS-2, ",
        "FFT-NS-i, NW-NS-1, NW-NS-2.", sep = "")
}

morie_mafft <- function(sequences, method = "FFT-NS-2", seq_type = NULL,
                        s_a = 0.06, s_op = 2.4, matrix = "normalized",
                        max_iterate = 16L) {
  mafft_alignment(sequences, method, seq_type, raw_matrix = NULL,
                  freqs = NULL, s_a = s_a, s_op = s_op, matrix = matrix,
                  max_iterate = max_iterate)
}
