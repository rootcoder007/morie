% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.mat}
\alias{.mat}
\title{.mat}
\usage{
.mat(x, name)
}
\arguments{
\item{x}{A matrix; passed to \code{nrow}.}
\item{name}{Passed to \code{sprintf}.}
}
\description{
A step of the explor_native implementation. Called by \code{explor}. See the file header for the source the module follows.
}
\value{
The value of \code{X}, as built in the body.
}
\references{
Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017)
"Curiosity-driven Exploration by Self-supervised Prediction", ICML,
arXiv:1705.05363. Eqs. 2-7.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.mat(V, V)
}
