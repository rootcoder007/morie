% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{build_chain}
\alias{build_chain}
\title{Build a complete chain from a list of entries}
\usage{
build_chain(entries, key = NULL, genesis = .GENESIS) 
}
\description{
Build a complete chain from a list of entries
}
\arguments{
\item{entries}{Argument \code{entries}; see Usage.}
\item{key}{Argument \code{key}; see Usage.}
\item{genesis}{Argument \code{genesis}; see Usage.}
}
\value{
A list with \code{hashes}, \code{head}, \code{n}, \code{head_hex}, \code{keyed}.
}
\keyword{internal}
\examples{
ch <- build_chain(list("alpha", "beta", "gamma"))
c(ch$n, nchar(ch$head_hex))
}
