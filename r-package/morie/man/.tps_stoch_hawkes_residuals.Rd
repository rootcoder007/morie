% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.tps_stoch_hawkes_residuals}
\alias{.tps_stoch_hawkes_residuals}
\title{Time-rescaling residuals for the exponential-kernel Hawkes fit;}
\usage{
.tps_stoch_hawkes_residuals(dt, mu, kappa, omega)
}
\arguments{
\item{dt}{Numeric; passed to \code{min}.}
\item{mu}{Numeric; combined arithmetically in the body.}
\item{kappa}{Numeric; combined arithmetically in the body.}
\item{omega}{Numeric; combined arithmetically in the body.}
}
\description{
Exp(1)-distributed when the model is correct (Ogata 1988).
}
\value{
The value of \code{res}, as built in the body.
}
\keyword{internal}
\examples{
.tps_stoch_hawkes_residuals(dt = c(1, 2, 3, 4, 5, 6, 7, 8), mu = c(1, 2, 3, 4, 5, 6, 7, 8), kappa = c(1, 2, 3, 4, 5, 6, 7, 8), omega = c(1, 2, 3, 4, 5, 6, 7, 8))
}
