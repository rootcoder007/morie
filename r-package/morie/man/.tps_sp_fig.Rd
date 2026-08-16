% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.tps_sp_fig}
\alias{.tps_sp_fig}
\title{Write one PNG under the caller-supplied fig_dir and return its path}
\usage{
.tps_sp_fig(fig_dir, name, draw, save_fig = TRUE, width = 1140, height = 620)
}
\arguments{
\item{fig_dir}{See Usage.}
\item{name}{See Usage.}
\item{draw}{See Usage.}
\item{save_fig}{Defaults to \code{TRUE}.}
\item{width}{Defaults to \code{1140}.}
\item{height}{Defaults to \code{620}.}
}
\description{
for the result\\\'s Figure line; with fig_dir = NULL nothing is written and the returned note says exactly that (no silent claims).
}
\value{
The value of \code{path}, as built in the body.
}
\keyword{internal}
