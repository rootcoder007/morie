% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.folds}
\alias{.folds}
\title{.folds}
\usage{
.folds(n, n_folds)
}
\arguments{
\item{n}{A count; the body uses it as \code{seq_len(...)}.}
\item{n_folds}{Coerced to integer by the body, with \code{as.integer}.}
}
\description{
Part of the tmldyn_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{lapply}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.folds(V, V)
}
