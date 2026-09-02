% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{tagRC_preference_vector}
\alias{tagRC_preference_vector}
\title{tagRC_preference_vector}
\usage{
tagRC_preference_vector(nodes, focus, weight = 0.9) 
}
\arguments{
\item{nodes}{Coerced to character by the body, with \code{as.character}.}
\item{focus}{Coerced to character by the body, with \code{as.character}.}
\item{weight}{Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{0.9}.}
}
\description{
Part of the tagRC_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{p}, \code{focus}, \code{mass}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
tagRC_preference_vector(V, V)
}
