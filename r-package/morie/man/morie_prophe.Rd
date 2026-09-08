% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_prophe}
\alias{morie_prophe}
\title{morie_prophe}
\usage{
morie_prophe(t, y, seasonalities = NULL, holidays = NULL, holiday_window = c(0, 
    0), ...) 
}
\arguments{
\item{t}{Passed to \code{prophe_additive_components}.}
\item{y}{Passed to \code{prophe_additive_components}.}
\item{seasonalities}{Passed to \code{prophe_additive_components}.}
\item{holidays}{Passed to \code{prophe_additive_components}.}
\item{holiday_window}{Passed to \code{prophe_additive_components}. Defaults to \code{c(0, 0)}.}
\item{...}{Passed through.}
}
\description{
Part of the prophe_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{prophe_additive_components}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
morie_prophe(V, V)
}
