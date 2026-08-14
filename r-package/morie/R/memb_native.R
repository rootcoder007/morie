# Sources: Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017)
# "Membership Inference Attacks Against Machine Learning Models", IEEE
# Symposium on Security and Privacy, 3-18.

.ghc_unif_int <- function(e, n) floor(.ghc_unif(e, n) * n)

.rng <- function(seed) .ghc_rng(seed)

logistic_trainer <- function(l2 = 1e-3, epochs = 300L, lr = 0.5, seed = 0) {
  function(X, y) {
    n <- length(X)
    if (n == 0L) stop("memb: cannot train on an empty dataset")
    d <- length(X[[1]])
    classes <- sort(unique(y))
    idx <- setNames(seq_along(classes) - 1L, classes)
    C <- length(classes)
    W <- matrix(0.0, C, d)
    b <- rep(0.0, C)
    for (ep in seq_len(as.integer(epochs))) {
      gW <- matrix(0.0, C, d)
      gb <- rep(0.0, C)
      for (i in seq_len(n)) {
        z <- rep(0.0, C)
        for (k in seq_len(C)) {
          s_ <- b[k]
          for (j in seq_len(d)) s_ <- s_ + W[k, j] * X[[i]][j]
          z[k] <- s_
        }
        mx <- max(z)
        ez <- exp(z - mx)
        ssum <- sum(ez)
        pr <- ez / ssum
        t <- idx[[y[i]]]
        for (k in seq_len(C)) {
          err <- (if (k == t + 1L) 1.0 else 0.0) - pr[k]
          gb[k] <- gb[k] + err
          gW[k, ] <- gW[k, ] + err * X[[i]]
        }
      }
      b <- b + lr * gb / n
      W <- W + lr * (gW / n - l2 * W)
    }
    classes_ref <- classes
    function(rows) {
      out <- matrix(0.0, length(rows), C)
      for (r in seq_along(rows)) {
        z <- rep(0.0, C)
        for (k in seq_len(C)) {
          s_ <- b[k]
          for (j in seq_len(d)) s_ <- s_ + W[k, j] * rows[[r]][j]
          z[k] <- s_
        }
        mx <- max(z)
        ez <- exp(z - mx)
        out[r, ] <- ez / sum(ez)
      }
      attr(out, "classes") <- classes_ref
      out
    }
  }
}

knn_trainer <- function(k = 1L, smoothing = 1e-3) {
  k <- as.integer(k)
  if (k < 1L) stop("memb: k must be >= 1")
  function(X, y) {
    if (length(X) == 0L) stop("memb: cannot train on an empty dataset")
    classes <- sort(unique(y))
    idx <- setNames(seq_along(classes) - 1L, classes)
    rows <- lapply(X, function(r) as.numeric(r))
    labs <- as.character(y)
    classes_ref <- classes
    function(query) {
      out <- matrix(0.0, length(query), length(classes))
      for (q in seq_along(query)) {
        qq <- query[[q]]
        d <- sapply(rows, function(rr) sum((rr - qq) ^ 2))
        ord <- order(d)
        votes <- rep(smoothing, length(classes))
        for (i in seq_len(k)) votes[idx[[labs[ord[i]]]] + 1L] <- votes[idx[[labs[ord[i]]]] + 1L] + 1.0
        out[q, ] <- votes / sum(votes)
      }
      attr(out, "classes") <- classes_ref
      out
    }
  }
}

attack_dataset <- function(model_predict, in_X, in_y, out_X, out_y) {
  rows <- list(); lab <- integer(0); cls <- character(0)
  if (length(in_X) > 0L) {
    if (is.matrix(model_predict) || is.numeric(model_predict)) {
      vecs <- model_predict
    } else {
      vecs <- model_predict(in_X)
    }
    for (i in seq_along(in_X)) {
      rows[[length(rows) + 1L]] <- as.numeric(vecs[i, ])
      lab <- c(lab, 1L)
      cls <- c(cls, as.character(in_y[i]))
    }
  }
  if (length(out_X) > 0L) {
    vecs2 <- model_predict(out_X)
    for (i in seq_along(out_X)) {
      rows[[length(rows) + 1L]] <- as.numeric(vecs2[i, ])
      lab <- c(lab, 0L)
      cls <- c(cls, as.character(out_y[i]))
    }
  }
  list(rows = rows, labels = lab, classes = cls)
}

synthesize <- function(target_predict, c, n_features, feature_values = NULL,
                        k_max = NULL, k_min = 1L, conf_min = 0.8,
                        iter_max = 1000L, rej_max = 10L, seed = 0) {
  n_features <- as.integer(n_features)
  if (n_features < 1L) stop("memb: n_features must be >= 1")
  if (conf_min <= 0 || conf_min >= 1) stop("memb: conf_min must lie in (0, 1)")
  if (k_min < 1L) stop("memb: k_min must be >= 1")
  k_max <- if (is.null(k_max)) n_features else as.integer(k_max)
  if (k_max < k_min) stop("memb: k_max must be at least k_min")
  e <- .rng(seed)
  vals <- if (is.null(feature_values)) lapply(seq_len(n_features), function(j) c(0.0, 1.0)) else feature_values
  if (length(vals) != n_features) stop("memb: feature_values must have one entry per feature")
  rand_record <- function(base = NULL, k = NULL) {
    if (is.null(base)) {
      x <- numeric(n_features)
      for (j in seq_len(n_features)) x[j] <- vals[[j]][.ghc_unif_int(e, length(vals[[j]])) + 1L]
      return(x)
    }
    x <- as.numeric(base)
    picks <- integer(0)
    while (length(picks) < min(k, n_features)) {
      p <- .ghc_unif_int(e, n_features)
      if (!(p %in% picks)) picks <- c(picks, p)
    }
    for (j in picks) {
      choices <- vals[[j]][vals[[j]] != x[j + 1L]]
      if (length(choices) == 0L) choices <- vals[[j]]
      x[j + 1L] <- choices[.ghc_unif_int(e, length(choices)) + 1L]
    }
    x
  }
  x <- rand_record()
  y_best <- 0.0
  x_best <- x
  j <- 0L
  k <- k_max
  for (it in seq_len(as.integer(iter_max))) {
    y <- as.numeric(target_predict(matrix(x, nrow = 1L))[1, ])
    if (c + 1L > length(y)) stop(sprintf("memb: class %r is outside the target's output vector", c))
    yc <- y[c + 1L]
    if (yc >= y_best) {
      if (yc > conf_min && c == which.max(y) - 1L) {
        if (.ghc_unif(e, 1L) < yc) return(x)
      }
      x_best <- x
      y_best <- yc
      j <- 0L
    } else {
      j <- j + 1L
      if (j > rej_max) {
        k <- max(k_min, ceiling(k / 2))
        j <- 0L
      }
    }
    x <- rand_record(x_best, k)
  }
  NULL
}

synthesize_marginals <- function(X, n, seed = 0) {
  if (length(X) == 0L) stop("memb: no data to take marginals from")
  e <- .rng(seed)
  d <- length(X[[1]])
  cols <- lapply(seq_len(d), function(j) sapply(X, function(row) row[j]))
  out <- matrix(0, as.integer(n), d)
  for (i in seq_len(as.integer(n))) for (j in seq_len(d))
    out[i, j] <- cols[[j]][.ghc_unif_int(e, length(cols[[j]])) + 1L]
  split(out, seq_len(nrow(out)))
}

synthesize_noisy <- function(X, fraction = 0.1, feature_values = NULL, seed = 0) {
  if (fraction < 0 || fraction > 1) stop("memb: fraction must lie in [0, 1]")
  e <- .rng(seed)
  d <- length(X[[1]])
  vals <- if (is.null(feature_values)) lapply(seq_len(d), function(j) sort(unique(sapply(X, function(row) row[j])))) else feature_values
  out <- lapply(X, function(row) {
    x <- as.numeric(row)
    for (j in seq_len(d)) {
      if (.ghc_unif(e, 1L) < fraction) {
        choices <- vals[[j]][vals[[j]] != x[j]]
        if (length(choices) == 0L) choices <- vals[[j]]
        x[j] <- choices[.ghc_unif_int(e, length(choices)) + 1L]
      }
    }
    x
  })
  out
}

precision_recall <- function(pred, truth) {
  tp <- sum(pred == 1L & truth == 1L)
  fp <- sum(pred == 1L & truth == 0L)
  fn <- sum(pred == 0L & truth == 1L)
  tn <- sum(pred == 0L & truth == 0L)
  n <- tp + fp + fn + tn
  list(precision = if (tp + fp > 0L) tp / (tp + fp) else NaN,
       recall = if (tp + fn > 0L) tp / (tp + fn) else NaN,
       accuracy = if (n > 0L) (tp + tn) / n else NaN,
       tp = tp, fp = fp, fn = fn, tn = tn)
}

.sorted_features <- function(vec, top = NULL) {
  s <- sort(vec, decreasing = TRUE)
  if (is.null(top)) s else s[seq_len(min(top, length(s)))]
}

memb <- function(target_predict, shadow_data, eval_in, eval_out,
                 train_fn = NULL, attack_train_fn = NULL, n_shadow = NULL,
                 sort_features = FALSE, threshold = 0.5) {
  if (is.null(train_fn)) train_fn <- logistic_trainer()
  if (is.null(attack_train_fn)) attack_train_fn <- logistic_trainer()
  specs <- as.list(shadow_data)
  if (!is.null(n_shadow)) specs <- specs[seq_len(min(as.integer(n_shadow), length(specs)))]
  if (length(specs) == 0L) stop("memb: at least one shadow model is needed")

  rows <- list(); labels <- integer(0); classes <- character(0)
  for (spec in specs) {
    tr_X <- spec[[1]]; tr_y <- spec[[2]]; te_X <- spec[[3]]; te_y <- spec[[4]]
    if (length(tr_X) == 0L) stop("memb: a shadow model has no training data")
    shadow <- train_fn(tr_X, tr_y)
    ds <- attack_dataset(shadow, tr_X, tr_y, te_X, te_y)
    rows <- c(rows, ds$rows); labels <- c(labels, ds$labels); classes <- c(classes, ds$classes)
  }
  if (length(rows) == 0L) stop("memb: the shadow models produced no attack data")

  per_class <- list()
  for (c in sort(unique(classes))) {
    idx <- which(classes == c)
    if (length(unique(labels[idx])) < 2L) next
    feats <- lapply(idx, function(t) if (sort_features) .sorted_features(rows[[t]]) else rows[[t]])
    per_class[[as.character(c)]] <- attack_train_fn(feats, labels[idx])
  }
  if (length(per_class) == 0L)
    stop("memb: no class had both in and out examples, so no attack model could be trained")

  eval_X <- c(as.list(eval_in[[1]]), as.list(eval_out[[1]]))
  eval_y <- c(as.character(eval_in[[2]]), as.character(eval_out[[2]]))
  truth <- c(rep(1L, length(eval_in[[1]])), rep(0L, length(eval_out[[1]])))
  outputs <- if (length(eval_X) > 0L) target_predict(do.call(rbind, lapply(eval_X, function(r) as.numeric(r)))) else matrix(0, 0, 0)
  scores <- numeric(0); preds <- integer(0)
  if (length(eval_X) > 0L) {
    for (i in seq_along(eval_X)) {
      c <- eval_y[i]
      model <- per_class[[c]]
      if (is.null(model)) { scores <- c(scores, NaN); preds <- c(preds, 0L); next }
      feat <- if (sort_features) .sorted_features(as.numeric(outputs[i, ])) else as.numeric(outputs[i, ])
      pr <- model(matrix(feat, nrow = 1L))[1, ]
      member <- if (length(pr) > 1L) pr[2L] else pr[1L]
      scores <- c(scores, member)
      preds <- c(preds, if (member >= threshold) 1L else 0L)
    }
  }
  metrics <- precision_recall(preds, truth)
  by_class <- list()
  for (c in sort(unique(eval_y))) {
    sel <- which(eval_y == c)
    if (length(sel) > 0L) by_class[[c]] <- precision_recall(preds[sel], truth[sel])
  }
  list(estimate = metrics, metrics = metrics, per_class = by_class,
       predictions = preds, scores = scores, truth = truth,
       n_shadow = length(specs), attack_train_size = length(rows),
       attack_classes = sort(names(per_class)), threshold = as.numeric(threshold),
       note = "the attack can only find a gap that exists: against a target that does not overfit, precision falls to the base rate (Shokri et al. 2017, section VII)",
       method = "shadow-trained membership inference (Shokri et al. 2017)")
}

cheatsheet <- function() {
  "memb: membership inference (Shokri et al. 2017). Black-box output vector in, member/non-member out. Train k SHADOW models on data distributed like the target's, where you DO know membership; their outputs on their own training data are labelled 'in' and on a disjoint test set 'out'; that labelled set trains the attack model -- one per output class, since the tell is class-conditional. Shadow data from Algorithm 1 synthesis against the target, from feature marginals, or from noisy real data. Metrics are precision and recall over members. The attack lives on the train/test gap: no overfitting, no attack."
}

membership_inference <- memb

morie_memb <- function(target_predict, shadow_data, eval_in, eval_out,
                      train_fn = NULL, attack_train_fn = NULL,
                      n_shadow = NULL, sort_features = FALSE,
                      threshold = 0.5) {
  memb(target_predict, shadow_data, eval_in, eval_out,
       train_fn = train_fn, attack_train_fn = attack_train_fn,
       n_shadow = n_shadow, sort_features = sort_features,
       threshold = threshold)
}
