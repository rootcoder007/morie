% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.psy_fixed_pool}
\alias{.psy_fixed_pool}
\title{.psy_fixed_pool}
\usage{
.psy_fixed_pool(yi, vi)
}
\arguments{
\item{yi}{Coerced to numeric by the body, with \code{as.numeric}.}
\item{vi}{Coerced to numeric by the body, with \code{as.numeric}.}
}
\description{
Part of the psycho_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{mu}, \code{Q}, \code{w}, \code{k}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.psy_fixed_pool(V, V)
}
