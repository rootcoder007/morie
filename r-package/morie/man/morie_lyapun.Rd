% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_lyapun}
\alias{morie_lyapun}
\title{morie_lyapun}
\usage{
morie_lyapun(y, embedding = NULL, tau = NULL, dt = 1, fit = NULL, min_sep = NULL, 
    max_steps = NULL, method = "rosenstein", k = NULL) 
}
\arguments{
\item{y}{Passed to \code{lyapunov_exponent}.}
\item{embedding}{Passed to \code{lyapunov_exponent}.}
\item{tau}{Passed to \code{lyapunov_exponent}.}
\item{dt}{Passed to \code{lyapunov_exponent}. Defaults to \code{1}.}
\item{fit}{Passed to \code{lyapunov_exponent}.}
\item{min_sep}{Passed to \code{lyapunov_exponent}.}
\item{max_steps}{Passed to \code{lyapunov_exponent}.}
\item{method}{Passed to \code{lyapunov_exponent}. Defaults to \code{"rosenstein"}.}
\item{k}{Passed to \code{lyapunov_exponent}.}
}
\description{
Part of the lyapun_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{lyapunov_exponent}.
}
\keyword{internal}
\examples{
r0 <- 3.9; x <- numeric(400); x[1] <- 0.4
for (i in 2:400) x[i] <- r0 * x[i - 1] * (1 - x[i - 1])
r <- morie_lyapun(x, embedding = 3, tau = 1)
str(r, max.level = 1)
}
