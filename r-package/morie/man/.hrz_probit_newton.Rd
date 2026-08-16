% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.hrz_probit_newton}
\alias{.hrz_probit_newton}
\title{.hrz_probit_newton}
\usage{
.hrz_probit_newton(D, Z, maxit = 50, tol = 1e-08)
}
\arguments{
\item{D}{Numeric; combined arithmetically in the body.}
\item{Z}{A matrix; passed to \code{ncol}.}
\item{maxit}{Defaults to \code{50}.}
\item{tol}{Defaults to \code{1e-08}.}
}
\description{
A step of the helpers_horowitz implementation. Called by \code{hrzs1}. See the file header for the source the module follows.
}
\value{
A vector, from \code{as.numeric}.
}
\keyword{internal}
\examples{
M <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2)
.hrz_probit_newton(M, M)
}
