% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{coherence}
\alias{coherence}
\title{coherence}
\usage{
coherence(A, rank = NULL) 
}
\arguments{
\item{A}{A matrix; passed to \code{as.matrix}.}
\item{rank}{Optional; may be \code{NULL}. Coerced to integer by the body, with \code{as.integer}.}
}
\description{
Part of the meglt_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{mu_row}, \code{mu_col}, \code{mu}, \code{rank}, \code{note}.
}
\keyword{internal}
\examples{
set.seed(1)
r <- coherence(matrix(rnorm(20), 5, 4), rank = 2)
str(r, max.level = 1)
}
