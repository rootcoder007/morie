% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.coerce_mat}
\alias{.coerce_mat}
\title{Build a row x col numeric matrix from a list-of-list input}
\usage{
.coerce_mat(m)
}
\arguments{
\item{m}{Optional; may be \code{NULL}. A vector; its length is taken.}
}
\description{
Part of the opnclp_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{do.call}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.coerce_mat(V)
}
