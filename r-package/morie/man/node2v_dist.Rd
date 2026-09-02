% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{node2v_dist}
\alias{node2v_dist}
\title{node2v_dist}
\usage{
node2v_dist(adj, t, x) 
}
\arguments{
\item{adj}{A vector; indexed elementwise.}
\item{t}{Coerced to character by the body, with \code{as.character}.}
\item{x}{Passed to \code{==}.}
}
\description{
Part of the node2v_native implementation; see the file header for the source it follows.
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
adj <- list(`0` = c("1", "3"), `1` = c("0", "2"),
            `2` = c("1", "3"), `3` = c("0", "2"))
c(node2v_dist(adj, "0", "0"), node2v_dist(adj, "0", "1"))
}
