% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{effective_sample_size}
\alias{effective_sample_size}
\title{Effective sample size}
\usage{
effective_sample_size(weights) 
}
\arguments{
\item{weights}{Numeric vector of positive weights.}
}
\description{
\code{(sum w)^2 / sum w^2}: how many particles are really contributing.
}
\value{
Numeric scalar.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
effective_sample_size(V)
}
