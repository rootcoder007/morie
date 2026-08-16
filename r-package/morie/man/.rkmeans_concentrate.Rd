% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.rkmeans_concentrate}
\alias{.rkmeans_concentrate}
\title{.rkmeans_concentrate}
\usage{
.rkmeans_concentrate(rows, cen, k, n_keep, penalty, huber_c, max_iter)
}
\arguments{
\item{rows}{A vector; its length is taken and its elements indexed.}
\item{cen}{A vector; indexed elementwise.}
\item{k}{A count; the body uses it as \code{seq_len(...)}.}
\item{n_keep}{A count; the body uses it as \code{seq_len(...)}.}
\item{penalty}{One of \code{"absolute"}, \code{"square"}.}
\item{huber_c}{Passed to \code{.rkmeans_phi}.}
\item{max_iter}{A count; the body uses it as \code{seq_len(...)}.}
}
\description{
Part of the rkmeans_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{list}.
}
\keyword{internal}
