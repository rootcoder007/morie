% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.new_influence_diag}
\alias{.new_influence_diag}
\title{.new_influence_diag}
\usage{
.new_influence_diag(h, cooks, dffits, dfbetas, covratio, influential, high_lev, high_cook)
}
\arguments{
\item{h}{See Usage.}
\item{cooks}{See Usage.}
\item{dffits}{See Usage.}
\item{dfbetas}{See Usage.}
\item{covratio}{See Usage.}
\item{influential}{A vector; its length is taken.}
\item{high_lev}{See Usage.}
\item{high_cook}{See Usage.}
}
\description{
A step of the diagnostics implementation. Called by \code{compute_influence}. See the file header for the source the module follows.
}
\value{
The value of \code{structure}.
}
\keyword{internal}
