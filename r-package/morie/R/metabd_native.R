# Sources: Kang, D. D., Li, F., Kirton, E., Thomas, A., Egan, R., An, H.
# & Wang, Z. (2019) "MetaBAT 2: an adaptive binning algorithm for
# robust and efficient genome reconstruction from metagenome
# assemblies", PeerJ 7, e7359, doi:10.7717/peerj.7359.
# Kang, D. D., Froula, J., Egan, R. & Wang, Z. (2015) "MetaBAT, an
# efficient tool for accurately reconstructing single genomes from
# complex microbial communities", PeerJ 3, e1165,
# doi:10.7717/peerj.1165.

.EPS <- 1e-12
.BASES <- strsplit("ACGT", "")[[1]]

.comp_map <- c(A = "T", C = "G", G = "C", T = "A")

tetranucleotide_frequency <- function(seq, kk = 4L, canonical = TRUE) {
  s <- toupper(as.character(seq))
  K <- as.integer(kk)
  if (K < 1L) stop("metabd: k must be at least 1")
  if (nchar(s) < K) stop("metabd: the sequence is shorter than k")
  chars <- strsplit(s, "")[[1]]
  counts <- list(); tot <- 0L
  for (i in seq_len(nchar(s) - K + 1L)) {
    m <- substring(s, i, i + K - 1L)
    if (any(!(strsplit(m, "")[[1]] %in% names(.comp_map)))) next
    if (canonical) {
      m_chars <- strsplit(m, "")[[1]]
      rc <- paste0(.comp_map[rev(m_chars)], collapse = "")
      m <- if (m < rc) m else rc
    }
    counts[[m]] <- if (is.null(counts[[m]])) 1L else counts[[m]] + 1L
    tot <- tot + 1L
  }
  if (tot == 0L) stop("metabd: no valid k-mers in the sequence")
  keys <- sort(names(counts))
  vec <- as.numeric(counts[keys]) / tot
  list(frequency = setNames(vec, keys), vector = vec, kmers = keys,
       n_kmers = tot, canonical = as.logical(canonical))
}

abundance_correlation <- function(cov_a, cov_b) {
  a <- as.numeric(cov_a); b <- as.numeric(cov_b)
  if (length(a) != length(b)) stop("metabd: the coverage vectors differ in length")
  if (length(a) < 2L)
    stop("metabd: abundance covariance needs at least 2 samples; with one sample only composition is informative")
  ma <- mean(a); mb <- mean(b)
  num <- sum((a - ma) * (b - mb))
  den <- sqrt(sum((a - ma) ^ 2) * sum((b - mb) ^ 2))
  list(correlation = if (den > .EPS) num / den else 0.0, n_samples = length(a))
}

length_weight <- function(length, l_min = 2500.0, l_ref = 100000.0) {
  L <- as.numeric(length)
  if (L <= 0.0) stop("metabd: the contig length must be positive")
  if (L < as.numeric(l_min))
    return(list(weight = 0.0, length = L, below_minimum = TRUE,
                note = "too short for a usable composition estimate"))
  w <- log(L / as.numeric(l_min)) / log(as.numeric(l_ref) / as.numeric(l_min))
  list(weight = min(max(w, 0.0), 1.0), length = L, below_minimum = FALSE)
}

composite_distance <- function(tnf_a, tnf_b, cov_a = NULL, cov_b = NULL,
                               len_a = NULL, len_b = NULL, w_abundance = 0.5) {
  a <- as.numeric(tnf_a); b <- as.numeric(tnf_b)
  if (length(a) != length(b)) stop("metabd: the composition vectors differ in length")
  d_tnf <- sqrt(sum((a - b) ^ 2))
  wa <- as.numeric(w_abundance)
  d_abd <- 0.0; usable <- FALSE
  if (!is.null(cov_a) && !is.null(cov_b) && length(cov_a) >= 2L) {
    r <- abundance_correlation(cov_a, cov_b)$correlation
    d_abd <- 1.0 - r
    usable <- TRUE
  }
  if (!usable) wa <- 0.0
  conf <- 1.0
  if (!is.null(len_a) && !is.null(len_b))
    conf <- min(length_weight(len_a)$weight, length_weight(len_b)$weight)
  d <- (1.0 - wa) * d_tnf + wa * d_abd
  list(distance = d, composition = d_tnf,
       abundance = if (usable) d_abd else NULL,
       abundance_usable = usable, confidence = conf,
       effective_weight = wa,
       note = "with a single sample the abundance term drops out automatically")
}

bin_contigs <- function(tnfs, coverages = NULL, lengths = NULL, threshold = 0.15,
                        min_bin_size = 200000.0) {
  T <- as.matrix(tnfs); n <- nrow(T)
  L <- if (is.null(lengths)) rep(1e5, n) else as.numeric(lengths)
  bins <- list(); assigned <- rep(FALSE, n)
  order <- order(-L)
  for (i in order) {
    if (assigned[i]) next
    cur <- c(i); assigned[i] <- TRUE
    for (j in order) {
      if (assigned[j]) next
      d <- composite_distance(T[i, ], T[j, ],
                              if (is.null(coverages)) NULL else coverages[[i]],
                              if (is.null(coverages)) NULL else coverages[[j]],
                              L[i], L[j])$distance
      if (d < as.numeric(threshold)) { cur <- c(cur, j); assigned[j] <- TRUE }
    }
    bins[[length(bins) + 1L]] <- cur
  }
  sizes <- sapply(bins, function(b) sum(L[b]))
  big_idx <- which(sizes >= as.numeric(min_bin_size))
  small_idx <- which(sizes < as.numeric(min_bin_size))
  big <- bins[big_idx]
  small <- unlist(bins[small_idx])
  list(estimate = big, bins = big, unbinned = sort(small),
       n_bins = length(big), n_unbinned = length(small),
       method = "adaptive composite binning; Kang et al. (2019)",
       note = "sub-threshold groups are UNBINNED, not reported as draft genomes")
}

purity_completeness <- function(bins, truth) {
  t <- as.character(truth)
  out <- list()
  for (b in bins) {
    labs <- t[b]
    if (length(labs) == 0L) next
    counts <- table(labs)
    dom <- names(which.max(counts))
    purity <- max(counts) / length(labs)
    total <- sum(t == dom)
    out[[length(out) + 1L]] <- list(dominant = dom, purity = purity,
                                     completeness = max(counts) / total,
                                     size = length(labs))
  }
  list(per_bin = out,
       mean_purity = if (length(out) > 0L) mean(sapply(out, function(x) x$purity)) else 0.0,
       mean_completeness = if (length(out) > 0L) mean(sapply(out, function(x) x$completeness)) else 0.0,
       note = "contamination and fragmentation are different failures")
}

.metabd_cheatsheet <- function() {
  "metabd: bin contigs into draft genomes from TWO signals -- tetranucleotide composition (available always, noisy on short contigs) and abundance covariance ACROSS SAMPLES (strong, but undefined with one sample). Earlier tools needed manual parameter tuning and degraded on poor assemblies; the contribution is an ADAPTIVE algorithm that removes the tuning. Confidence must scale with contig LENGTH, since discarding short contigs discards most of the assembly. Purity and completeness are separate failures and are reported separately."
}

metabat2 <- bin_contigs
metagenome_binning <- bin_contigs

morie_metabd <- function(tnfs, coverages = NULL, lengths = NULL,
                        threshold = 0.15, min_bin_size = 200000.0) {
  bin_contigs(tnfs, coverages = coverages, lengths = lengths,
              threshold = threshold, min_bin_size = min_bin_size)
}
