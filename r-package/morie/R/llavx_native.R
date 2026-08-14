# Sources: Liu, H., Li, C., Wu, Q. & Lee, Y. J. (2023) "Visual Instruction
# Tuning", NeurIPS 2023, arXiv:2304.08485.
# Radford, A. et al. (2021) "Learning Transferable Visual Models From
# Natural Language Supervision", ICML 2021, PMLR 139, 8748-8763,
# arXiv:2103.00020.
# Li, J., Li, D., Savarese, S. & Hoi, S. (2023) "BLIP-2", ICML 2023,
# PMLR 202, 19730-19742, arXiv:2301.12597.

.KINDS <- c("conversation", "detailed_description", "complex_reasoning")

symbolic_representation <- function(captions, boxes) {
  caps <- as.character(captions)
  bx <- as.list(boxes)
  if (length(caps) == 0L && length(bx) == 0L)
    stop("llavx: an image with no captions and no boxes has no symbolic representation")
  lines <- as.list(caps)
  for (rec in bx) {
    name <- as.character(rec[[1]])
    x <- as.numeric(rec[[2]]); y <- as.numeric(rec[[3]])
    w <- as.numeric(rec[[4]]); h <- as.numeric(rec[[5]])
    lines[[length(lines) + 1L]] <- sprintf("%s: [%.3f, %.3f, %.3f, %.3f]", name, x, y, w, h)
  }
  list(text = paste(unlist(lines), collapse = "\n"),
       n_captions = length(caps),
       n_boxes = length(bx),
       note = "the generator is LANGUAGE-ONLY; the image itself never reaches it")
}

instruction_prompt <- function(symbolic, kind = "conversation") {
  if (!(kind %in% .KINDS))
    stop(sprintf("llavx: kind must be one of %s, got %r",
                 paste(.KINDS, collapse = ", "), kind))
  ask <- switch(kind,
    conversation = "Ask and answer questions about this image as if you can see it.",
    detailed_description = "Describe this image in detail.",
    complex_reasoning = "Give a question requiring step-by-step reasoning about this image, and answer it."
  )
  list(prompt = paste0(symbolic[["text"]], "\n\n", ask), kind = kind)
}

project_patches <- function(patch_features, W, b = NULL) {
  F <- lapply(patch_features, function(r) as.numeric(r))
  d_out <- nrow(W)
  if (ncol(W) != length(F[[1]]))
    stop(sprintf("llavx: the projection expects %d features but got %d",
                 ncol(W), length(F[[1]])))
  bb <- if (is.null(b)) rep(0.0, d_out) else as.numeric(b)
  out <- vector("list", length(F))
  for (i in seq_along(F)) {
    f <- F[[i]]
    row <- numeric(d_out)
    for (o in seq_len(d_out)) {
      s <- bb[o]
      wrow <- as.numeric(W[o, ])
      for (j in seq_along(f)) s <- s + wrow[j] * f[j]
      row[o] <- s
    }
    out[[i]] <- row
  }
  out
}

build_sequence <- function(visual_tokens, text_embeddings) {
  V <- lapply(visual_tokens, function(r) as.numeric(r))
  T <- lapply(text_embeddings, function(r) as.numeric(r))
  if (length(V) > 0L && length(T) > 0L && length(V[[1]]) != length(T[[1]]))
    stop(sprintf("llavx: visual tokens are %d-dimensional but text embeddings are %d -- the projection target is wrong",
                 length(V[[1]]), length(T[[1]])))
  estimate <- c(V, T)
  list(estimate = estimate, sequence = estimate,
       n_visual = length(V), n_text = length(T),
       method = "visual instruction tuning; Liu, Li, Wu & Lee (2023)",
       note = "projected patches ARE tokens -- no cross-attention layers are introduced")
}

training_stage <- function(stage) {
  s <- as.integer(stage)
  if (!(s %in% c(1L, 2L)))
    stop(sprintf("llavx: the stage must be 1 or 2, got %r", stage))
  if (s == 1L) {
    list(stage = 1L, trainable = list("projection"),
         frozen = list("vision_encoder", "language_model"),
         data = "image-caption pairs",
         note = "align the spaces before tuning anything on them")
  } else {
    list(stage = 2L, trainable = list("projection", "language_model"),
         frozen = list("vision_encoder"),
         data = "GPT-4 generated instruction-following data",
         note = "tuning the language model first would tune it against features that do not yet mean anything")
  }
}

cheatsheet <- function() {
  "llavx: instruction tuning works in language and lacked MULTIMODAL data, so generate it with a LANGUAGE-ONLY GPT-4 fed a SYMBOLIC image -- captions and boxes. The image never reaches the generator, which is what makes the pipeline possible and also caps it: what the captions omit cannot be asked about. Architecture is deliberately thin: ONE projection matrix into the word-embedding space, projected patches used as tokens, no cross-attention. Stage 1 trains only the projection; stage 2 adds the language model."
}

visualinstruction <- build_sequence
llava_visual_chat <- build_sequence

morie_llavx <- function(captions = NULL, boxes = NULL, kind = "conversation",
                       stage = NULL) {
  if (!is.null(stage)) return(training_stage(stage))
  sym <- symbolic_representation(captions, boxes)
  instruction_prompt(sym, kind = kind)
}
