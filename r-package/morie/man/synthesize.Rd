% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{synthesize}
\alias{synthesize}
\title{synthesize}
\usage{
synthesize(target_predict, c, n_features, feature_values = NULL, k_max = NULL, k_min = 1L, conf_min = 0.8, iter_max = 1000L, rej_max = 10L, seed = 0)
}
\arguments{
\item{target_predict}{Accepted by the signature and not used anywhere in the body.}
\item{c}{Numeric; combined arithmetically in the body.}
\item{n_features}{A count; the body uses it as \code{seq_len(...)}.}
\item{feature_values}{The body requires: memb: feature_values must have one entry per feature.}
\item{k_max}{Optional; may be \code{NULL}. Coerced to integer by the body, with \code{as.integer}.}
\item{k_min}{Numeric; passed to \code{max}. Defaults to \code{1L}.}
\item{conf_min}{The body requires: memb: conf_min must lie in (0, 1). Defaults to \code{0.8}.}
\item{iter_max}{Coerced to integer by the body, with \code{as.integer}. Defaults to \code{1000L}.}
\item{rej_max}{Passed to \code{>}. Defaults to \code{10L}.}
\item{seed}{Passed to \code{.memb_rng}. Defaults to \code{0}.}
}
\description{
Part of the memb_native implementation; see the file header for the source it follows.
}
\value{
Nothing; the function is called for its effect.
}
\keyword{internal}
\examples{
set.seed(1)
target <- function(X) {
  z <- cbind(rowSums(X), -rowSums(X))
  p <- exp(z - apply(z, 1, max))
  p / rowSums(p)
}
r <- synthesize(target, c = 0L, n_features = 3,
                feature_values = list(c(-1, 0, 1), c(-1, 0, 1),
                                      c(-1, 0, 1)))
str(r, max.level = 1)
}
