% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{epsilon_data_independent}
\alias{epsilon_data_independent}
\title{Section 3.2: 4 T gamma^2 + 2 gamma sqrt(2 T ln(1/delta)). The}
\usage{
epsilon_data_independent(T, gamma, delta)
}
\arguments{
\item{T}{See Usage.}
\item{gamma}{See Usage.}
\item{delta}{See Usage.}
}
\description{
composition of T queries, each (2 gamma, 0)-DP, without looking at the votes.
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
epsilon_data_independent(T = 5L, gamma = 0.5, delta = 0.5)
}
