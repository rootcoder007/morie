% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.propinf_zero_grads}
\alias{.propinf_zero_grads}
\title{.propinf_zero_grads}
\usage{
.propinf_zero_grads(model)
}
\arguments{
\item{model}{A list; the body reads \code{$phis}, \code{$psis}, \code{$rho} from it.}
}
\description{
Part of the propinf_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{phis}, \code{psis}, \code{rho}.
}
\keyword{internal}
\examples{
D <- data.frame(x = c(1, 2, 3, 4), y = c(2, 4, 5, 9))
.propinf_zero_grads(D)
}
