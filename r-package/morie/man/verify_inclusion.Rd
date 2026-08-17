% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{verify_inclusion}
\alias{verify_inclusion}
\title{Recompute the root from a leaf and an audit path}
\usage{
verify_inclusion(leaf, index, size, path, root)
}
\description{
Recompute the root from a leaf and an audit path
}
\value{
A list with \code{root}, \code{root_hex}, \code{valid}, \code{path_used}.
}
\keyword{internal}
\examples{
leaves <- list("a", "b", "c", "d", "e")
root <- merkle_root(leaves)
pr <- inclusion_proof(leaves, index = 2)
good <- verify_inclusion("c", 2, 5, pr$path, root)
bad <- verify_inclusion("x", 2, 5, pr$path, root)
c(good, bad)
}
