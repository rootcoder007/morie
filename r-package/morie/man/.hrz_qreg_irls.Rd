% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.hrz_qreg_irls}
\alias{.hrz_qreg_irls}
\title{.hrz_qreg_irls}
\usage{
.hrz_qreg_irls(X, y, tau = 0.5, maxit = 50, tol = 1e-06)
}
\arguments{
\item{X}{A matrix; passed to \code{t}.}
\item{y}{Numeric; combined arithmetically in the body.}
\item{tau}{Numeric; combined arithmetically in the body. Defaults to \code{0.5}.}
\item{maxit}{Passed to \code{:}. Defaults to \code{50}.}
\item{tol}{Passed to \code{<}. Defaults to \code{1e-06}.}
}
\description{
A step of the helpers_horowitz implementation. Called by \code{hrzc1}, \code{hrzq1}. See the file header for the source the module follows.
}
\value{
A vector, from \code{as.numeric}.
}
\keyword{internal}
