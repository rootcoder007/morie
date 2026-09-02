# Segment Anything: promptable segmentation as a pre-training task.
# Sources: Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C.,
# Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y.,
# Dollar, P. & Girshick, R. (2023) "Segment Anything", *Proceedings
# of the IEEE/CVF International Conference on Computer Vision (ICCV
# 2023)*, 4015-4026, arXiv:2304.02643. Sec. 2 (the promptable
# segmentation task, and the requirement that the output be a
# reasonable mask for at least one object even when the prompt is
# ambiguous), Sec. 3 (the three constraints -- flexible prompting,
# amortised real-time computation, ambiguity-awareness; the image
# encoder run once per image; sparse prompts as positional encodings
# summed with learned per-type embeddings and dense mask prompts
# embedded by convolution and summed with the image embedding; ~50 ms
# per prompt in a browser), and Sec. 5 (SA-1B: over 1B masks on 11M
# licensed, privacy-respecting images). Dosovitskiy, A. et al.
# (2021) "An Image is Worth 16x16 Words", *ICLR 2021*,
# arXiv:2010.11929. The ViT image encoder. He, K., Chen, X., Xie, S.,
# Li, Y., Dollar, P. & Girshick, R. (2022) "Masked Autoencoders Are
# Scalable Vision Learners", *CVPR 2022*, 16000-16009,
# arXiv:2111.06377. The MAE pre-training used for it.

.SAMSEG_EPS <- 1e-12
.SAMSEG_TYPES <- c("foreground", "background", "box_tl", "box_br")

#' .samseg_mat
#'
#' A step of the samseg_native implementation. Called by \code{encode_mask_prompt}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param x A matrix; passed to \code{as.matrix}.
#' @return Nothing; this branch always raises.
#' @export
.samseg_mat <- function(x) {
  if (is.matrix(x)) return(x)
  if (is.numeric(x)) return(as.matrix(x))
  if (is.list(x)) {
    n <- length(x)
    if (is.list(x[[1]])) {
      d <- length(x[[1]])
      M <- matrix(0, nrow = n, ncol = d)
      for (i in seq_len(n)) M[i, ] <- as.numeric(x[[i]])
      return(M)
    }
    return(matrix(as.numeric(unlist(x)), nrow = n))
  }
  stop("samseg: expected a matrix-like input")
}

#' .samseg_pos_enc
#'
#' A step of the samseg_native implementation. Called by \code{encode_box_prompt}, \code{encode_point_prompt}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param x Coerced to numeric by the body, with \code{as.numeric}.
#' @param y Coerced to numeric by the body, with \code{as.numeric}.
#' @param dim A count; the body uses it as \code{numeric(...)}. Defaults to \code{8}.
#' @param scale Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{1}.
#' @return The value of \code{out}, as built in the body.
#' @export
.samseg_pos_enc <- function(x, y, dim = 8, scale = 1.0) {
  out <- numeric(dim)
  for (j in seq_len(as.integer(dim) %/% 2L)) {
    f <- (2.0 ^ (j - 1L)) * pi * as.numeric(scale)
    out[2L * j - 1L] <- sin(f * as.numeric(x))
    out[2L * j]      <- cos(f * as.numeric(y))
  }
  out
}

#' encode_point_prompt
#'
#' A step of the samseg_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param points Iterated over elementwise, with \code{lapply}.
#' @param labels Coerced to integer by the body, with \code{as.integer}.
#' @param dim Passed to \code{.samseg_pos_enc}. Defaults to \code{8}.
#' @param type_embeddings Defaults to \code{NULL}.
#' @return A list with \code{tokens}, \code{n_prompts}, \code{sparse}, \code{note}.
#' @export
encode_point_prompt <- function(points, labels, dim = 8,
                                type_embeddings = NULL) {
  P <- lapply(points, function(p) c(as.numeric(p[1]), as.numeric(p[2])))
  L <- as.integer(labels)
  if (length(P) != length(L))
    stop("samseg: ", length(P), " points but ", length(L), " labels")
  if (any(!(L == 0L | L == 1L)))
    stop("samseg: a point label must be 1 (foreground) or 0 (background)")
  te <- type_embeddings
  if (is.null(te)) te <- list()
  out <- list()
  for (i in seq_along(P)) {
    e <- .samseg_pos_enc(P[[i]][1], P[[i]][2], dim)
    name <- if (L[i] == 1L) "foreground" else "background"
    t <- te[[name]]
    if (is.null(t)) t <- rep(0, length(e))
    t <- as.numeric(t)
    if (length(t) != length(e))
      stop("samseg: the type embedding has the wrong width")
    out[[i]] <- e + t
  }
  M <- do.call(rbind, out)
  list(tokens = M, n_prompts = length(out), sparse = TRUE,
       note = "a background click at the same place is a DIFFERENT token, by the type embedding")
}

#' encode_box_prompt
#'
#' A step of the samseg_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param box Coerced to numeric by the body, with \code{as.numeric}.
#' @param dim Passed to \code{.samseg_pos_enc}. Defaults to \code{8}.
#' @param type_embeddings Defaults to \code{NULL}.
#' @return A list with \code{tokens}, \code{n_prompts}, \code{sparse}.
#' @export
encode_box_prompt <- function(box, dim = 8, type_embeddings = NULL) {
  b <- as.numeric(box)
  x0 <- b[1]
  y0 <- b[2]
  x1 <- b[3]
  y1 <- b[4]
  if (x1 <= x0 || y1 <= y0)
    stop("samseg: the box is empty or inverted")
  te <- type_embeddings
  if (is.null(te)) te <- list()
  a <- .samseg_pos_enc(x0, y0, dim)
  bb <- .samseg_pos_enc(x1, y1, dim)
  ta <- te[["box_tl"]]
  if (is.null(ta)) ta <- rep(0, length(a))
  tb <- te[["box_br"]]
  if (is.null(tb)) tb <- rep(0, length(bb))
  ta <- as.numeric(ta)
  tb <- as.numeric(tb)
  M <- rbind(a + ta, bb + tb)
  list(tokens = M, n_prompts = 2L, sparse = TRUE)
}

#' encode_mask_prompt
#'
#' A step of the samseg_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param mask Passed to \code{.samseg_mat}.
#' @param image_embedding Passed to \code{.samseg_mat}.
#' @param weight Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{1}.
#' @return A list with \code{embedding}, \code{sparse}, \code{note}.
#' @export
encode_mask_prompt <- function(mask, image_embedding, weight = 1.0) {
  M <- .samseg_mat(mask)
  E <- .samseg_mat(image_embedding)
  if (nrow(M) != nrow(E) || ncol(M) != ncol(E))
    stop("samseg: the mask prompt is ", nrow(M), "x", ncol(M),
         " but the image embedding is ", nrow(E), "x", ncol(E))
  w <- as.numeric(weight)
  embed <- E + w * M
  list(embedding = embed, sparse = FALSE,
       note = "summed, so the decoder input shape is unchanged")
}

#' amortised_cost
#'
#' A step of the samseg_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param encoder_ms Coerced to numeric by the body, with \code{as.numeric}.
#' @param decoder_ms Coerced to numeric by the body, with \code{as.numeric}.
#' @param n_prompts Coerced to integer by the body, with \code{as.integer}.
#' @return A list with \code{total_ms}, \code{per_prompt_ms}, \code{naive_ms}, \code{speedup}, \code{interactive}, \code{note}.
#' @export
amortised_cost <- function(encoder_ms, decoder_ms, n_prompts) {
  e <- as.numeric(encoder_ms)
  d <- as.numeric(decoder_ms)
  P <- as.integer(n_prompts)
  if (P < 1L)
    stop("samseg: at least one prompt is needed")
  if (e <= 0.0 || d <= 0.0)
    stop("samseg: the timings must be positive")
  total <- e + P * d
  list(total_ms = total, per_prompt_ms = total / P,
       naive_ms = P * (e + d),
       speedup = P * (e + d) / total,
       interactive = d < 100.0,
       note = "the image embedding is computed once and reused")
}

#' promptable_segment
#'
#' A step of the samseg_native implementation. Called by \code{morie_samseg}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param image_embedding Passed to \code{decoder}.
#' @param prompt_tokens Passed to \code{decoder}.
#' @param decoder The body requires: samseg: the decoder returned no mask; the task requires a valid mask for ANY prompt.
#' @param multimask Coerced to logical by the body, with \code{as.logical}. Defaults to \code{TRUE}.
#' @return A list with \code{estimate}, \code{masks}, \code{n_masks}, \code{multimask}, \code{method}, \code{note}.
#' @export
promptable_segment <- function(image_embedding, prompt_tokens, decoder,
                               multimask = TRUE) {
  masks <- decoder(image_embedding, prompt_tokens, multimask)
  if (length(masks) == 0L)
    stop("samseg: the decoder returned no mask; the task requires a valid mask for ANY prompt")
  list(estimate = masks[[1]], masks = masks, n_masks = length(masks),
       multimask = as.logical(multimask),
       method = "promptable segmentation; Kirillov et al. (2023)",
       note = "a valid mask for any prompt, and for an ambiguous prompt a valid mask for at least one intended object")
}

#' morie_samseg
#'
#' A step of the samseg_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param image_embedding Passed to \code{promptable_segment}.
#' @param prompt_tokens Passed to \code{promptable_segment}.
#' @param decoder Passed to \code{promptable_segment}.
#' @param multimask Passed to \code{promptable_segment}. Defaults to \code{TRUE}.
#' @return The value of \code{promptable_segment}.
#' @export
morie_samseg <- function(image_embedding, prompt_tokens, decoder,
                         multimask = TRUE) {
  promptable_segment(image_embedding, prompt_tokens, decoder,
                     multimask = multimask)
}

sam_segment <- promptable_segment
samsegment <- promptable_segment
segmentanything <- promptable_segment

#' .samseg_cheatsheet
#'
#' A step of the samseg_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @return A character value.
#' @export
.samseg_cheatsheet <- function() {
  paste("samseg: the task is 'return a VALID mask for any prompt,",
        "and for an AMBIGUOUS prompt a valid mask for at least one",
        "intended object' -- which is what makes it usable as",
        "pre-training and for zero-shot transfer by prompting.",
        "Three constraints force the architecture: flexible prompts,",
        "amortised real-time use, ambiguity-awareness. So a heavy",
        "image encoder runs ONCE per image and a light prompt",
        "encoder plus mask decoder run per prompt (~50 ms). Sparse",
        "prompts are positional encodings plus a learned PER-TYPE",
        "embedding; dense mask prompts are SUMMED with the image",
        "embedding.")
}
