% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{verify_chain}
\alias{verify_chain}
\title{Verify a chain, returning the FIRST bad index (not just a boolean)}
\usage{
verify_chain(entries, hashes, key = NULL, genesis = .GENESIS)
}
\description{
Verify a chain, returning the FIRST bad index (not just a boolean)
}
\value{
A list with \code{estimate}, \code{intact}, \code{first_bad}, \code{verified_through}, \code{n}, \code{method}, \code{note}.
}
\keyword{internal}
