# Mask R-CNN: instance segmentation by adding a mask branch.
# Sources: He, K., Gkioxari, G., Dollar, P. & Girshick, R. (2017)
# "Mask R-CNN", Proceedings of the IEEE International Conference
# on Computer Vision (ICCV 2017), 2980-2988,
# doi:10.1109/ICCV.2017.322, arXiv:1703.06870 -- extending
# Faster R-CNN with a mask branch in parallel with classification
# and box regression; RoIPool's coarse spatial quantisation; the
# RoIAlign layer that preserves exact spatial locations; the
# multi-task loss L = L_cls + L_box + L_mask; and the per-class
# binary masks with a per-pixel sigmoid, decoupling mask and class
# prediction, against the per-pixel softmax which makes classes
# compete. Ren, S., He, K., Girshick, R. & Sun, J. (2015) "Faster
# R-CNN: Towards Real-Time Object Detection with Region Proposal
# Networks", NeurIPS 2015, 91-99, arXiv:1506.01497 -- the detector
# being extended.
#
# Native implementation mirroring Python morie.fn.masrcn exactly:
# same RoIPool with two roundings and a max, same RoIAlign with
# bilinear sampling at the bin centres, same shift measurement in
# feature and input pixels, same per-pixel sigmoid binary loss with
# the per-pixel softmax as the worse alternative, same unweighted
# three-term sum.

.masrcn_eps <- 1e-12

#' .masrcn_bilinear
#'
#' A step of the masrcn_native implementation. Called by \code{roi_align}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param F A matrix; indexed by row and column.
#' @param y Numeric; combined arithmetically in the body.
#' @param x Numeric; combined arithmetically in the body.
#' @return A numeric value.
#' @export
.masrcn_bilinear <- function(F, y, x) {
  h <- nrow(F)
  w <- ncol(F)
  y <- min(max(as.numeric(y), 0.0), h - 1.0)
  x <- min(max(as.numeric(x), 0.0), w - 1.0)
  y0 <- as.integer(floor(y))
  x0 <- as.integer(floor(x))
  y1 <- min(y0 + 1L, h - 1L)
  x1 <- min(x0 + 1L, w - 1L)
  dy <- y - y0
  dx <- x - x0
  F[y0 + 1L, x0 + 1L] * (1 - dy) * (1 - dx) +
    F[y1 + 1L, x0 + 1L] * dy * (1 - dx) +
    F[y0 + 1L, x1 + 1L] * (1 - dy) * dx +
    F[y1 + 1L, x1 + 1L] * dy * dx
}

#' roi_pool
#'
#' A step of the masrcn_native implementation. Called by \code{alignment_error}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param features A matrix; passed to \code{as.matrix}.
#' @param box A vector; indexed elementwise.
#' @param out_size Coerced to integer by the body, with \code{as.integer}. Defaults to \code{2L}.
#' @param stride Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{1}.
#' @return A list with \code{pooled}, \code{quantised_box}, \code{quantisation_shift}, \code{caveat}.
#' @export
roi_pool <- function(features, box, out_size = 2L, stride = 1.0) {
  F <- as.matrix(features)
  storage.mode(F) <- "double"
  y0 <- as.numeric(box[1]) / as.numeric(stride)
  x0 <- as.numeric(box[2]) / as.numeric(stride)
  y1 <- as.numeric(box[3]) / as.numeric(stride)
  x1 <- as.numeric(box[4]) / as.numeric(stride)
  qy0 <- as.integer(floor(y0))
  qx0 <- as.integer(floor(x0))
  qy1 <- as.integer(floor(y1))
  qx1 <- as.integer(floor(x1))
  if (qy1 <= qy0 || qx1 <= qx0)
    stop("masrcn: the box collapsed under quantisation, which is itself the problem")
  n <- as.integer(out_size)
  bh <- (qy1 - qy0) / n
  bw <- (qx1 - qx0) / n
  out <- matrix(0, n, n)
  for (i in 0:(n - 1L)) {
    for (j in 0:(n - 1L)) {
      a0 <- qy0 + as.integer(floor(i * bh))
      a1 <- max(a0 + 1L, qy0 + as.integer(floor((i + 1L) * bh)))
      b0 <- qx0 + as.integer(floor(j * bw))
      b1 <- max(b0 + 1L, qx0 + as.integer(floor((j + 1L) * bw)))
      a1c <- min(a1, nrow(F))
      b1c <- min(b1, ncol(F))
      if (a0 < a1c && b0 < b1c) {
        vals <- as.numeric(F[(a0 + 1L):a1c, (b0 + 1L):b1c])
        out[i + 1L, j + 1L] <- if (length(vals)) max(vals) else 0
      } else {
        out[i + 1L, j + 1L] <- 0
      }
    }
  }
  list(pooled = out,
       quantised_box = c(qy0, qx0, qy1, qx1),
       quantisation_shift = c(y0 - qy0, x0 - qx0),
       caveat = "the box AND the bins are rounded to the feature grid")
}

#' roi_align
#'
#' A step of the masrcn_native implementation. Called by \code{morie_masrcn}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param features A matrix; passed to \code{as.matrix}.
#' @param box A vector; indexed elementwise.
#' @param out_size Coerced to integer by the body, with \code{as.integer}. Defaults to \code{2L}.
#' @param stride Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{1}.
#' @param samples Coerced to integer by the body, with \code{as.integer}. Defaults to \code{2L}.
#' @return A list with \code{pooled}, \code{exact_box}, \code{samples_per_bin}, \code{note}.
#' @export
roi_align <- function(features, box, out_size = 2L, stride = 1.0,
                      samples = 2L) {
  F <- as.matrix(features)
  storage.mode(F) <- "double"
  y0 <- as.numeric(box[1]) / as.numeric(stride)
  x0 <- as.numeric(box[2]) / as.numeric(stride)
  y1 <- as.numeric(box[3]) / as.numeric(stride)
  x1 <- as.numeric(box[4]) / as.numeric(stride)
  if (y1 <= y0 || x1 <= x0)
    stop("masrcn: the box has non-positive extent")
  n <- as.integer(out_size)
  s <- as.integer(samples)
  bh <- (y1 - y0) / n
  bw <- (x1 - x0) / n
  out <- matrix(0, n, n)
  for (i in 0:(n - 1L)) {
    for (j in 0:(n - 1L)) {
      acc <- numeric(s * s)
      k <- 0L
      for (a in 0:(s - 1L)) {
        for (b in 0:(s - 1L)) {
          yy <- y0 + bh * (i + (a + 0.5) / s)
          xx <- x0 + bw * (j + (b + 0.5) / s)
          k <- k + 1L
          acc[k] <- .masrcn_bilinear(F, yy, xx)
        }
      }
      out[i + 1L, j + 1L] <- sum(acc) / length(acc)
    }
  }
  list(pooled = out, exact_box = c(y0, x0, y1, x1),
       samples_per_bin = s * s,
       note = "no quantisation of the box or the bins")
}

#' alignment_error
#'
#' A step of the masrcn_native implementation. Called by \code{morie_masrcn}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param features Passed to \code{roi_pool}.
#' @param box Passed to \code{roi_pool}.
#' @param out_size Passed to \code{roi_pool}. Defaults to \code{2L}.
#' @param stride Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{1}.
#' @return A list with \code{feature_shift}, \code{input_pixel_shift}, \code{stride}, \code{note}.
#' @export
alignment_error <- function(features, box, out_size = 2L, stride = 1.0) {
  p <- roi_pool(features, box, out_size, stride)
  dy <- p$quantisation_shift[1]
  dx <- p$quantisation_shift[2]
  list(feature_shift = c(dy, dx),
       input_pixel_shift = c(dy * as.numeric(stride),
                              dx * as.numeric(stride)),
       stride = as.numeric(stride),
       note = paste("a sub-pixel error on the feature map is a ",
                    "several-pixel error in the image at stride 16 ",
                    "or 32", sep = ""))
}

#' mask_loss
#'
#' A step of the masrcn_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param logits A matrix; passed to \code{as.matrix}.
#' @param target A matrix; passed to \code{as.matrix}.
#' @param decoupled A flag; the body branches on it. Defaults to \code{TRUE}.
#' @return A list, whose contents depend on the branch taken; across the branches its names are \code{loss}, \code{kind}, \code{note}, \code{caveat}.
#' @export
mask_loss <- function(logits, target, decoupled = TRUE) {
  L <- as.matrix(logits)
  storage.mode(L) <- "double"
  T <- as.matrix(target)
  storage.mode(T) <- "double"
  if (nrow(L) != nrow(T) || ncol(L) != ncol(T))
    stop("masrcn: the logits and target differ in shape")
  tot <- 0.0
  m <- 0L
  if (decoupled) {
    for (i in seq_len(nrow(L)))
      for (j in seq_len(ncol(L))) {
        p <- if (L[i, j] > -700) 1.0 / (1.0 + exp(-L[i, j])) else 0.0
        p <- min(max(p, .masrcn_eps), 1.0 - .masrcn_eps)
        tot <- tot - (T[i, j] * log(p) + (1 - T[i, j]) * log(1 - p))
        m <- m + 1L
      }
    list(loss = tot / m, kind = "per-pixel sigmoid",
         note = "classes do not compete; the class branch decides the category")
  } else {
    flat <- as.numeric(L)
    mx <- max(flat)
    z <- sum(exp(flat - mx))
    for (i in seq_len(nrow(L)))
      for (j in seq_len(ncol(L))) {
        p <- exp(L[i, j] - mx) / z
        tot <- tot - T[i, j] * log(max(p, .masrcn_eps))
        m <- m + 1L
      }
    list(loss = tot / m, kind = "per-pixel softmax",
         caveat = "classes COMPETE, so a pixel assigned to one is evidence against another")
  }
}

#' multitask_loss
#'
#' A step of the masrcn_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param l_cls Coerced to numeric by the body, with \code{as.numeric}.
#' @param l_box Coerced to numeric by the body, with \code{as.numeric}.
#' @param l_mask Coerced to numeric by the body, with \code{as.numeric}.
#' @return A list with \code{total}, \code{cls}, \code{box}, \code{mask}, \code{note}.
#' @export
multitask_loss <- function(l_cls, l_box, l_mask) {
  list(total = as.numeric(l_cls) + as.numeric(l_box) + as.numeric(l_mask),
       cls = as.numeric(l_cls), box = as.numeric(l_box),
       mask = as.numeric(l_mask),
       note = "an unweighted sum, which the decoupling permits")
}

maskrcnn <- roi_align
mask_rcnn_segmentation <- roi_align

#' .masrcn_cheatsheet
#'
#' A step of the masrcn_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @return A character value.
#' @export
.masrcn_cheatsheet <- function() {
  paste("masrcn: Faster R-CNN plus a THIRD branch predicting a ",
        "binary mask per RoI. Two details carry it. RoIPool ",
        "QUANTISES twice -- box and bins -- which is fine for a ",
        "box and a several-pixel misalignment for a mask at stride ",
        "16 or 32; RoIAlign removes both roundings and samples ",
        "bilinearly. And the mask is DECOUPLED from the class: K ",
        "binary masks with a per-pixel SIGMOID, loss on the ",
        "ground-truth class only, because a per-pixel softmax ",
        "makes classes compete. The decoupling is what lets the ",
        "losses simply add.", sep = "")
}

#' morie_masrcn
#'
#' A step of the masrcn_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param features Passed to \code{roi_align}.
#' @param box Passed to \code{roi_align}.
#' @param out_size Passed to \code{roi_align}. Defaults to \code{2L}.
#' @param stride Passed to \code{roi_align}. Defaults to \code{1}.
#' @param samples Passed to \code{roi_align}. Defaults to \code{2L}.
#' @return A list with \code{roi}, \code{alignment}.
#' @export
morie_masrcn <- function(features, box, out_size = 2L, stride = 1.0,
                        samples = 2L) {
  r <- roi_align(features, box, out_size, stride, samples)
  list(roi = r, alignment = alignment_error(features, box, out_size, stride))
}
