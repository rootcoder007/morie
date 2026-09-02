% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{merkle_root}
\alias{merkle_root}
\title{RFC 6962 Merkle Tree Hash}
\usage{
merkle_root(leaves) 
}
\description{
RFC 6962 Merkle Tree Hash
}
\arguments{
\item{leaves}{Argument \code{leaves}; see Usage.}
}
\value{
The value of \code{sha256}.
}
\keyword{internal}
\examples{
root <- merkle_root(list("a", "b", "c", "d"))
length(root) == 32L
}
