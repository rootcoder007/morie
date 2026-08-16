% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{split_sample}
\alias{split_sample}
\title{split_sample}
\usage{
split_sample(n, V = 10L, seed = 0L)
}
\arguments{
\item{n}{A count; the body uses it as \code{seq_len(...)}.}
\item{V}{A count; the body uses it as \code{seq_len(...)}. Defaults to \code{10L}.}
\item{seed}{Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{0L}.}
}
\description{
Part of the tldapar_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{estimation}, \code{training}, \code{V}.
}
\keyword{internal}
