% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.smoopt_dual_objective}
\alias{.smoopt_dual_objective}
\title{Dual objective of the C-SVM, in the Platt sign convention so it}
\usage{
.smoopt_dual_objective(alpha, y, K)
}
\arguments{
\item{alpha}{See Usage.}
\item{y}{See Usage.}
\item{K}{See Usage.}
}
\description{
is directly comparable to svmopt.dual_objective up to the sign of the threshold term -- the threshold does not enter the dual itself.
}
\value{
A numeric value.
}
\keyword{internal}
