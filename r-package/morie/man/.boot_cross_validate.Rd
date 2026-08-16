% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.boot_cross_validate}
\alias{.boot_cross_validate}
\title{K-fold cross-validation}
\usage{
.boot_cross_validate(X, y, model_fn, score_fn, n_folds = 10L, stratify = NULL, groups = NULL, seed = 42L)
}
\arguments{
\item{X}{Numeric design matrix.}
\item{y}{Response vector.}
\item{model_fn}{Function \code{(X_train, y_train) -> model}.}
\item{score_fn}{Function \code{(y_true, y_pred) -> scalar}.}
\item{n_folds}{Number of folds.}
\item{stratify}{Optional stratification vector.}
\item{groups}{Optional grouping vector (no group split across folds).}
\item{seed}{Random seed.}
}
\description{
Supports plain, stratified, and grouped variants. Bootstrap-flavoured K-fold cross-validation (internal). Lower-level CV used by \[repeated_cv()\] / \[leave_one_out_cv()\]. Public CV with `(fit_fn, predict_fn, X, y, ...)` signature lives in \[cross_validate()\] (validation.R). Renamed to avoid the symbol collision that R CMD check surfaced as unused-arg notes.
}
\value{
A \code{morie_cv_result}.
}
\keyword{internal}
