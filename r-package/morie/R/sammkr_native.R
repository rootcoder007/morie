# Three masks, one gradient: SAM's answer to ambiguity.
# Sources: Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C.,
# Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y.,
# Dollar, P. & Girshick, R. (2023) "Segment Anything", *ICCV 2023*,
# 4015-4026, arXiv:2304.02643. Sec. 3, "Resolving ambiguity": that
# with one output the model will AVERAGE multiple valid masks given
# an ambiguous prompt; the modification to predict multiple output
# masks for a single prompt; that 3 mask outputs is sufficient for
# most common cases since nested masks are often at most three deep
# (whole, part and subpart); that during training only the MINIMUM
# loss over masks is backpropagated; and that the model predicts a
# confidence score (estimated IoU) for each mask so they can be
# ranked.

.SAMMKR_EPS <- 1e-12
.SAMMKR_NESTING <- c("whole", "part", "subpart")

#' .sammkr_flat
#'
#' A step of the sammkr_native implementation. Called by \code{iou}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param m A matrix; the body checks with \code{is.matrix}.
#' @return A vector, from \code{as.numeric}.
#' @export
.sammkr_flat <- function(m) {
  if (is.matrix(m)) return(as.numeric(m))
  if (is.list(m)) {
    out <- c()
    for (r in m) out <- c(out, as.numeric(r))
    return(out)
  }
  as.numeric(m)
}

#' average_of_valid_masks
#'
#' A step of the sammkr_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param masks Iterated over elementwise, with \code{lapply}.
#' @return A list with \code{mask}, \code{ambiguous_fraction}, \code{n_averaged}, \code{note}.
#' @export
average_of_valid_masks <- function(masks) {
  F <- lapply(masks, .sammkr_flat)
  if (length(F) == 0L)
    stop("sammkr: no masks given")
  n <- length(F[[1]])
  if (any(vapply(F, length, integer(1)) != n))
    stop("sammkr: the masks differ in size")
  avg <- numeric(n)
  for (i in seq_len(n)) {
    s <- 0
    for (f in F) s <- s + f[i]
    avg[i] <- s / length(F)
  }
  frac <- sum(avg > 0.05 & avg < 0.95) / n
  list(mask = avg, ambiguous_fraction = frac,
       n_averaged = length(F),
       note = "pixels strictly between 0 and 1 belong to no single valid interpretation")
}

#' iou
#'
#' A step of the sammkr_native implementation. Called by \code{rank_masks}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param a Passed to \code{.sammkr_flat}.
#' @param b Passed to \code{.sammkr_flat}.
#' @param threshold Defaults to \code{0.5}.
#' @return A numeric value.
#' @export
iou <- function(a, b, threshold = 0.5) {
  x <- as.numeric(.sammkr_flat(a) > threshold)
  y <- as.numeric(.sammkr_flat(b) > threshold)
  if (length(x) != length(y))
    stop("sammkr: the masks differ in size")
  inter <- sum(x > 0 & y > 0)
  uni <- sum(x > 0 | y > 0)
  if (uni == 0) return(1.0)
  inter / uni
}

#' min_loss_over_masks
#'
#' A step of the sammkr_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param predictions A vector; its length is taken.
#' @param target See Usage.
#' @param loss_fn Accepted by the signature and not used anywhere in the body.
#' @return A list with \code{loss}, \code{index}, \code{losses}, \code{mean_loss}, \code{gap}, \code{note}.
#' @export
min_loss_over_masks <- function(predictions, target, loss_fn) {
  if (length(predictions) == 0L)
    stop("sammkr: no predictions given")
  losses <- vapply(predictions, function(p) as.numeric(loss_fn(p, target)),
                   numeric(1))
  j <- which.min(losses)
  mean <- sum(losses) / length(losses)
  list(loss = losses[j], index = j, losses = losses,
       mean_loss = mean, gap = mean - losses[j],
       note = sprintf("only output %d receives gradient; the others are free to specialise elsewhere", j))
}

#' whole_part_subpart
#'
#' A step of the sammkr_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param masks A vector; its length is taken.
#' @param target_hierarchy Accepted by the signature and not used anywhere in the body.
#' @return A list with \code{assignment}, \code{sizes}, \code{nested}, \code{note}.
#' @export
whole_part_subpart <- function(masks, target_hierarchy = NULL) {
  if (length(masks) != 3L)
    stop("sammkr: the paper's argument is about THREE outputs (whole, part, subpart), got ",
         length(masks))
  flat <- lapply(masks, .sammkr_flat)
  sizes <- vapply(flat, function(v) sum(v > 0.5), integer(1))
  order <- order(-sizes)
  named <- list()
  for (rank in seq_len(3)) {
    named[[.SAMMKR_NESTING[rank]]] <- order[rank] - 1L
  }
  # nested: each strictly smaller mask's positive indices are a subset
  # of the previous (larger) mask's positive indices.
  nested <- TRUE
  for (r in seq_len(2)) {
    smaller <- which(flat[[order[r + 1]]] > 0.5)
    larger  <- which(flat[[order[r]]] > 0.5)
    if (!all(smaller %in% larger)) {
      nested <- FALSE
      break
    }
  }
  list(assignment = named, sizes = sizes, nested = nested,
       note = "nested masks are often at most three deep")
}

#' rank_masks
#'
#' A step of the sammkr_native implementation. Called by \code{morie_sammkr}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param masks A vector; its length is taken.
#' @param predicted_iou Coerced to numeric by the body, with \code{as.numeric}.
#' @param target Defaults to \code{NULL}.
#' @return The value of \code{out}, as built in the body.
#' @export
rank_masks <- function(masks, predicted_iou, target = NULL) {
  p <- as.numeric(predicted_iou)
  if (length(p) != length(masks))
    stop("sammkr: ", length(masks), " masks but ", length(p),
         " predicted IoUs")
  order <- order(-p)
  out <- list(order = order, best = order[1], predicted_iou = p)
  if (!is.null(target)) {
    true <- vapply(masks, function(m) iou(m, target), numeric(1))
    best_true <- which.max(true)
    out$true_iou <- true
    out$best_true <- best_true
    out$correct <- order[1] == best_true
    out$calibration_error <- sum(abs(p - true)) / length(p)
    out$regret <- true[best_true] - true[order[1]]
  }
  out$estimate <- order[1]
  out$method <- "multi-mask output with IoU ranking; Kirillov et al. (2023)"
  out$note <- paste("the score is a LEARNED estimate, so its error is",
                    "reported rather than assumed away")
  out
}

#' morie_sammkr
#'
#' A step of the sammkr_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param masks See Usage.
#' @param predicted_iou See Usage.
#' @param target Defaults to \code{NULL}.
#' @return The value of \code{rank_masks}.
#' @export
morie_sammkr <- function(masks, predicted_iou, target = NULL) {
  rank_masks(masks, predicted_iou, target = target)
}

sam_multi_mask_rank <- rank_masks
sammultimask <- rank_masks

#' .sammkr_cheatsheet
#'
#' A step of the sammkr_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @return A character value.
#' @export
.sammkr_cheatsheet <- function() {
  paste("sammkr: one output forces the model to AVERAGE the valid",
        "masks of an ambiguous prompt -- a blur that answers nobody.",
        "So predict THREE, because segmentation nesting is usually at",
        "most three deep: whole, part, subpart. During training",
        "backprop only the MINIMUM loss, which is what makes the",
        "three specialise instead of collapsing into one (the mean",
        "would collapse them). At inference there is no ground",
        "truth, so the model predicts its own IoU per mask to rank",
        "them -- a learned estimate, so report its calibration error.")
}
