% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.hrz_nw_loo}
\alias{.hrz_nw_loo}
\title{.hrz_nw_loo}
\usage{
.hrz_nw_loo(z, y, h)
}
\arguments{
\item{z}{A matrix; indexed by row and column.}
\item{y}{A matrix; passed to \code{\%*\%}.}
\item{h}{Numeric; combined arithmetically in the body.}
}
\description{
A step of the helpers_horowitz implementation. Called by \code{hrzp1}, \code{morie_ichimura}. See the file header for the source the module follows.
}
\value{
A vector, from \code{as.numeric}.
}
\keyword{internal}
