% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.explor_softmax}
\alias{.explor_softmax}
\title{.explor_softmax}
\usage{
.explor_softmax(z)
}
\arguments{
\item{z}{Numeric; passed to \code{max}.}
}
\description{
A step of the explor_native implementation. Called by \code{explor}. See the file header for the source the module follows.
}
\value{
A numeric value.
}
\references{
Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017)
"Curiosity-driven Exploration by Self-supervised Prediction", ICML,
arXiv:1705.05363. Eqs. 2-7.
}
\keyword{internal}
