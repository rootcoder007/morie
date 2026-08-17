% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{gp_predict}
\alias{gp_predict}
\title{Posterior mean and sd at theta}
\usage{
gp_predict(fit, theta)
}
\description{
Posterior mean and sd at theta
}
\value{
A vector.
}
\keyword{internal}
\examples{
set.seed(1)
design <- sobol_sequence(16, 1)
values <- as.numeric(sin(4 * design[, 1]))
fit <- gp_fit(design, values)
pr <- gp_predict(fit, 0.5)
c(length(pr), pr["sd"] >= 0)
}
