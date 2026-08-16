# Sources: Huang, Y., Lv, T., Cui, L., Lu, Y. & Wei, F. (2022)
# "LayoutLMv3: Pre-training for Document AI with Unified Text and Image
# Masking", ACM MM '22, 4083-4091, doi:10.1145/3503161.3548112,
# arXiv:2204.08387 (unified masking of text and image patches with
# discrete visual tokens, word-patch alignment, linear patch embedding);
# Dosovitskiy, A. et al. (2021) "An Image is Worth 16x16 Words", ICLR
# 2021, arXiv:2010.11929 (linear patch embedding); Bao, H., Dong, L.,
# Piao, S. & Wei, F. (2022) "BEiT: BERT Pre-Training of Image
# Transformers", ICLR 2022, arXiv:2106.08254 (discrete visual tokens as
# targets).
#
# Native implementation mirroring Python morie.fn.ocrwit exactly: the
# same box normalisation onto a 1000-grid, the same segment-level union
# of per-word boxes, the same block-masking recipe for text and image
# units, the same patch-of-box mapping, and the same unmasked-only
# word-patch alignment labels.

#' ocrwit_normalise_bbox
#'
#' A step of the ocrwit_native implementation. Called by \code{ocrwit_patch_of_box}, \code{ocrwit_segment_layout_boxes}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param box A vector; indexed elementwise.
#' @param width Coerced to numeric by the body, with \code{as.numeric}.
#' @param height Coerced to numeric by the body, with \code{as.numeric}.
#' @param scale Coerced to integer by the body, with \code{as.integer}. Defaults to \code{1000}.
#' @return A vector, from \code{c}.
#' @export
ocrwit_normalise_bbox <- function(box, width, height, scale = 1000) {
  x0 <- as.numeric(box[[1]]); y0 <- as.numeric(box[[2]])
  x1 <- as.numeric(box[[3]]); y1 <- as.numeric(box[[4]])
  W <- as.numeric(width); H <- as.numeric(height)
  if (W <= 0 || H <= 0)
    stop("ocrwit: the page dimensions must be positive")
  if (x1 < x0 || y1 < y0) stop("ocrwit: the box is inverted")
  s <- as.integer(scale)
  clamp <- function(v) max(0L, min(s, as.integer(round(v))))
  c(clamp(x0 / W * s), clamp(y0 / H * s),
    clamp(x1 / W * s), clamp(y1 / H * s))
}

#' ocrwit_segment_layout_boxes
#'
#' A step of the ocrwit_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param boxes Coerced to list by the body, with \code{as.list}.
#' @param segment_ids Coerced to list by the body, with \code{as.list}.
#' @param width Passed to \code{ocrwit_normalise_bbox}.
#' @param height Passed to \code{ocrwit_normalise_bbox}.
#' @param scale Passed to \code{ocrwit_normalise_bbox}. Defaults to \code{1000}.
#' @return A list with \code{segment_boxes}, \code{per_token}, \code{n_segments}, \code{note}.
#' @export
ocrwit_segment_layout_boxes <- function(boxes, segment_ids, width, height,
                                         scale = 1000) {
  segs <- as.list(segment_ids); B <- as.list(boxes)
  if (length(segs) != length(B))
    stop("ocrwit: ", length(B), " boxes but ", length(segs),
         " segment ids")
  nms <- unique(as.character(unlist(segs)))
  seg_box <- list()
  for (s in nms) {
    idx <- which(vapply(segs, function(x) identical(as.character(x), s),
                        logical(1)))
    nb <- lapply(idx, function(i) ocrwit_normalise_bbox(B[[i]], width,
                                                       height, scale))
    seg_box[[s]] <- c(min(vapply(nb, `[`, numeric(1), 1)),
                      min(vapply(nb, `[`, numeric(1), 2)),
                      max(vapply(nb, `[`, numeric(1), 3)),
                      max(vapply(nb, `[`, numeric(1), 4)))
  }
  per_token <- lapply(segs, function(s) seg_box[[as.character(s)]])
  list(segment_boxes = seg_box, per_token = per_token,
       n_segments = length(seg_box),
       note = "one box per segment, cheaper than per word and closer to the document's structure")
}

#' ocrwit_mask_units
#'
#' A step of the ocrwit_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param n_units Coerced to integer by the body, with \code{as.integer}.
#' @param rate Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{0.3}.
#' @param seed Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{0}.
#' @param block Coerced to integer by the body, with \code{as.integer}. Defaults to \code{1}.
#' @return A list with \code{masked}, \code{kept}, \code{rate}, \code{block}, \code{note}.
#' @export
ocrwit_mask_units <- function(n_units, rate = 0.3, seed = 0, block = 1) {
  n <- as.integer(n_units)
  r <- as.numeric(rate)
  if (n < 1L) stop("ocrwit: there is nothing to mask")
  if (!(r > 0 && r < 1)) stop("ocrwit: the mask rate must lie in (0,1)")
  rng <- .ghc_rng(as.numeric(seed))
  b <- max(1L, as.integer(block))
  masked <- integer(0)
  target <- max(1L, as.integer(round(n * r)))
  guard <- 0L
  while (length(unique(masked)) < target && guard < 1000L * n) {
    u <- .ghc_unif(rng, 1L)
    s <- as.integer(u * n) %% n
    blk <- seq.int(s, min(n - 1L, s + b - 1L))
    masked <- c(masked, blk)
    guard <- guard + 1L
  }
  masked <- sort(unique(masked))
  kept <- sort(setdiff(seq_len(n) - 1L, masked))
  list(masked = masked, kept = kept,
       rate = length(masked) / as.numeric(n), block = b,
       note = "the same recipe for both modalities, which is the unification")
}

#' ocrwit_patch_of_box
#'
#' A step of the ocrwit_native implementation. Called by \code{morie_ocrwit}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param box Passed to \code{ocrwit_normalise_bbox}.
#' @param width Passed to \code{ocrwit_normalise_bbox}.
#' @param height Passed to \code{ocrwit_normalise_bbox}.
#' @param patch_grid Coerced to integer by the body, with \code{as.integer}. Defaults to \code{14}.
#' @return A vector, from \code{sort}.
#' @export
ocrwit_patch_of_box <- function(box, width, height, patch_grid = 14) {
  g <- as.integer(patch_grid)
  bb <- ocrwit_normalise_bbox(box, width, height, g)
  x0 <- bb[1]; y0 <- bb[2]; x1 <- bb[3]; y1 <- bb[4]
  r0 <- min(y0, g - 1L)
  r1 <- min(max(y1, y0 + 1L), g)
  c0 <- min(x0, g - 1L)
  c1 <- min(max(x1, x0 + 1L), g)
  out <- c()
  for (r in r0:(r1 - 1L)) {
    for (c in c0:(c1 - 1L)) {
      out <- c(out, r * g + c)
    }
  }
  sort(unique(out))
}

#' morie_ocrwit
#'
#' A step of the ocrwit_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param text_boxes A vector; its length is taken and its elements indexed.
#' @param masked_patches Passed to \code{unlist}.
#' @param width Passed to \code{ocrwit_patch_of_box}.
#' @param height Passed to \code{ocrwit_patch_of_box}.
#' @param patch_grid Passed to \code{ocrwit_patch_of_box}. Defaults to \code{14}.
#' @param masked_text Passed to \code{unlist}. Defaults to \code{list()}.
#' @return A list with \code{estimate}, \code{labels}, \code{patches}, \code{n_examples}, \code{positive_rate}, \code{method}, \code{note}.
#' @export
morie_ocrwit <- function(text_boxes, masked_patches, width, height,
                         patch_grid = 14, masked_text = list()) {
  mp <- as.integer(unlist(masked_patches))
  mt <- as.integer(unlist(masked_text))
  labels <- list(); covered <- list()
  for (i in seq_along(text_boxes)) {
    if (i - 1L %in% mt) next
    ps <- ocrwit_patch_of_box(text_boxes[[i]], width, height, patch_grid)
    covered[[as.character(i - 1L)]] <- ps
    labels[[as.character(i - 1L)]] <- as.integer(any(ps %in% mp))
  }
  if (length(labels) == 0L)
    stop("ocrwit: every text token is masked, so the alignment objective has no examples")
  lab_vals <- vapply(labels, identity, integer(1))
  list(estimate = labels, labels = labels, patches = covered,
       n_examples = length(labels),
       positive_rate = sum(lab_vals) / as.numeric(length(labels)),
       method = "word-patch alignment; Huang, Lv, Cui, Lu & Wei (2022)",
       note = "unmasked words only -- a masked word would leak its own reconstruction target")
}

ocrwit_word_patch_alignment <- morie_ocrwit
layoutlmv3 <- morie_ocrwit
ocr_wit_layout <- morie_ocrwit
ocrwitlayout <- morie_ocrwit

#' ocrwit_cheatsheet
#'
#' A step of the ocrwit_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @return A character value.
#' @export
ocrwit_cheatsheet <- function() {
  paste("ocrwit: document models pre-trained text and image with ",
        "DIFFERENT objectives, giving two spaces and a bridge. ",
        "LayoutLMv3 makes them symmetric -- mask and reconstruct ",
        "text tokens, mask and reconstruct image patches as DISCRETE ",
        "tokens -- so one encoder learns one space. Linear patch ",
        "embeddings, so no CNN backbone or detector. WORD-PATCH ",
        "ALIGNMENT binds them: for an UNMASKED word, predict whether ",
        "its patch was masked, which is the only objective that ",
        "forces the model to know where a word sits. Layout is ",
        "SEGMENT-level 2D position.")
}
