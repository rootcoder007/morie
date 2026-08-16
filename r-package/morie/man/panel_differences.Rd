% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{panel_differences}
\alias{panel_differences}
\title{Post-minus-pre panel differences}
\usage{
panel_differences(Y, event_time)
}
\arguments{
\item{Y}{Balanced n-by-T panel.}
\item{event_time}{\code{H}, in 1-based period numbers.}
}
\description{
The differenced outcome \code{Delta_i} of eq. (13.7) under the block-adoption design.
}
\value{
Numeric vector of length n.
}
\keyword{internal}
