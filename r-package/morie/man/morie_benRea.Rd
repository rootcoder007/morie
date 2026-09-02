% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_benRea}
\alias{morie_benRea}
\title{morie_benRea}
\usage{
morie_benRea(...) 
}
\arguments{
\item{...}{Passed through.}
}
\description{
A step of the benRea_native implementation. No other function in the package calls it. See the file header for the source the module follows.
}
\value{
The value of \code{ner_decode}.
}
\keyword{internal}
\examples{
emissions <- list(c(2, 0, 0), c(0, 2, 0), c(0, 0, 2))
r <- morie_benRea(emissions, types = list("PER"))
r$path[[2]]
}
