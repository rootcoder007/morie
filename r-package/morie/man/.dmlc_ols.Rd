% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.dmlc_ols}
\alias{.dmlc_ols}
\title{Per-arm OLS outcome regression; robust to rank-deficiency and thin}
\usage{
.dmlc_ols(X, idx, y, te, p)
}
\arguments{
\item{X}{A matrix; indexed by row and column.}
\item{idx}{A vector; its length is taken.}
\item{y}{A vector; indexed elementwise.}
\item{te}{A vector; its length is taken.}
\item{p}{Numeric; combined arithmetically in the body.}
}
\description{
arms Part of the dml_clustered implementation; see the file header for the source it follows.
}
\value{
A vector, from \code{as.numeric}.
}
\keyword{internal}
\examples{
.dmlc_ols(X = c(1, 2, 3, 4, 5, 6, 7, 8), idx = 5L, y = c(1, 2, 3, 4, 5, 6, 7, 8), te = c(1, 2, 3, 4, 5, 6, 7, 8), p = 0.5)
}
