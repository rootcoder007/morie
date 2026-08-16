% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{miprgr_branch_and_bound}
\alias{miprgr_branch_and_bound}
\title{miprgr_branch_and_bound}
\usage{
miprgr_branch_and_bound(A, b, c, integer_vars, maximise = TRUE, prune = TRUE, max_nodes = 5000, solver = "simplex")
}
\arguments{
\item{A}{See Usage.}
\item{b}{See Usage.}
\item{c}{A vector; its length is taken.}
\item{integer_vars}{Coerced to integer by the body, with \code{as.integer}.}
\item{maximise}{A flag; the body branches on it. Defaults to \code{TRUE}.}
\item{prune}{A flag; the body branches on it. Defaults to \code{TRUE}.}
\item{max_nodes}{Coerced to integer by the body, with \code{as.integer}. Defaults to \code{5000}.}
\item{solver}{Defaults to \code{"simplex"}.}
}
\description{
Part of the miprgr_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{estimate}, \code{value}, \code{x}, \code{feasible}, \code{nodes}, \code{pruned}, \code{pruning}, \code{max_list_length}, \code{root_bound}, \code{truncated}, \code{method}, \code{note}.
}
\references{
  Land, A. H. & Doig, A. G. (1960) Econometrica 28(3), 497-520.
  Dakin, R. J. (1965) The Computer Journal 8(3), 250-255.
Base R only.
}
\keyword{internal}
\examples{
miprgr_branch_and_bound(A = matrix(c(1, 2, 3, 4, 5, 6), nrow = 2), b = 5L, c = matrix(c(1, 2, 3, 4, 5, 6), nrow = 2), integer_vars = matrix(c(1, 2, 3, 4, 5, 6), nrow = 2))
}
