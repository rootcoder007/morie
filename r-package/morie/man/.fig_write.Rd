% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.fig_write}
\alias{.fig_write}
\title{Render one figure to figures/<name>.pdf (and optionally .png).}
\usage{
.fig_write(fig_dir, name, draw, png_too = FALSE, width = 8, height = 6)
}
\arguments{
\item{fig_dir}{Passed to \code{file.path}.}
\item{name}{Passed to \code{warning}.}
\item{draw}{Accepted by the signature and not used anywhere in the body.}
\item{png_too}{A flag; the body branches on it. Defaults to \code{FALSE}.}
\item{width}{Numeric; combined arithmetically in the body. Defaults to \code{8}.}
\item{height}{Numeric; combined arithmetically in the body. Defaults to \code{6}.}
}
\description{
Returns the relative paths written, or character(0) if the draw function errored (the half-written file is removed and a warning names the figure).
}
\value{
The value of \code{wrote}, as built in the body.
}
\keyword{internal}
