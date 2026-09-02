% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_causscd}
\alias{morie_causscd}
\title{morie_causscd}
\usage{
morie_causscd(Y, treated, t_post, zeta = NULL) 
}
\arguments{
\item{Y}{Passed to \code{sdid}.}
\item{treated}{Passed to \code{sdid}.}
\item{t_post}{Passed to \code{sdid}.}
\item{zeta}{Passed to \code{sdid}.}
}
\description{
A step of the causscd_native implementation. No other function in the package calls it. See the file header for the source the module follows.
}
\value{
The value of \code{p}, as built in the body.
}
\keyword{internal}
\examples{
set.seed(2)
Y <- matrix(rnorm(80, 10), 8, 10)
Y[7:8, 7:10] <- Y[7:8, 7:10] + 3
treated <- c(rep(0L, 6), 1L, 1L)
r <- morie_causscd(Y, treated, t_post = 6)
str(r, max.level = 1)
}
