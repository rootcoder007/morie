% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.Phi}
\alias{.Phi}
\title{Pnorm is the standard normal CDF. 2 * pnorm(z) - 1 = .erf(z/sqrt(2))}
\usage{
.Phi(z)
}
\arguments{
\item{z}{See Usage.}
}
\description{
Using pnorm is exactly the closed form below.
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.Phi(V)
}
