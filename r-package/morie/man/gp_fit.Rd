% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{gp_fit}
\alias{gp_fit}
\title{Fit a GP emulator to a log-likelihood ensemble}
\usage{
gp_fit(design, values, nugget = NULL, lengthscale = NULL, kernel = "sqexp", 
    tau2 = NULL) 
}
\description{
Wilkinson (2014) Sec. 2.1: quadratic mean, conjugate improper \eqn{1/tau^2} prior on \eqn{(beta, tau^2)}, plug-in MLE length-scale.
}
\arguments{
\item{design}{Argument \code{design}; see Usage.}
\item{values}{Argument \code{values}; see Usage.}
\item{nugget}{Argument \code{nugget}; see Usage.}
\item{lengthscale}{Argument \code{lengthscale}; see Usage.}
\item{kernel}{Argument \code{kernel}; see Usage.}
\item{tau2}{Argument \code{tau2}; see Usage.}
}
\value{
A list with \code{design}, \code{values}, \code{beta}, \code{tau2}, \code{lengthscale}, \code{kernel}, \code{nugget}, \code{chol}, \code{Ainv_r}, \code{Ainv_H}, \code{H}, \code{HtAinvH_chol}, \code{n}, \code{q}, \code{dim}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
gp_fit(V, V)
}
