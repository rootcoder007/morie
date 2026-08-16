% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{prophe_additive_components}
\alias{prophe_additive_components}
\title{prophe_additive_components}
\usage{
prophe_additive_components(t, y, seasonalities = NULL, holidays = NULL, holiday_window = c(0, 0), ...)
}
\arguments{
\item{t}{Passed to \code{morie_prphet}.}
\item{y}{Passed to \code{morie_prphet}.}
\item{seasonalities}{Optional; may be \code{NULL}. A vector; its length is taken.}
\item{holidays}{Optional; may be \code{NULL}. A vector; its length is taken.}
\item{holiday_window}{A vector; indexed elementwise. Defaults to \code{c(0, 0)}.}
\item{...}{Passed through.}
}
\description{
Part of the prophe_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{estimate}, \code{components}, \code{total}, \code{fitted}, \code{residual}, \code{reconstruction_error}, \code{reconstructs}, \code{coef}, \code{changepoints}, \code{sigma}, \code{n}, \code{component_names}, \code{method}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
prophe_additive_components(V, V)
}
