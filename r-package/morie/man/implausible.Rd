% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{implausible}
\alias{implausible}
\title{Wilkinson implausibility rule, eq. (3)}
\usage{
implausible(fit, theta, threshold = 10, n_sd = 3)
}
\description{
Wilkinson implausibility rule, eq. (3)
}
\value{
The value of \code{as.logical}.
}
\keyword{internal}
\examples{
set.seed(1)
design <- sobol_sequence(16, 1)
values <- as.numeric(sin(4 * design[, 1]))
fit <- gp_fit(design, values)
is.logical(implausible(fit, 0.5, threshold = 1))
}
