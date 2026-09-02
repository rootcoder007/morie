% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{mtdrl_TabularHistoryAgent}
\alias{mtdrl_TabularHistoryAgent}
\title{mtdrl_TabularHistoryAgent}
\usage{
mtdrl_TabularHistoryAgent(n_arms, epsilon = 0.1, optimistic = 1) 
}
\arguments{
\item{n_arms}{Coerced to integer by the body, with \code{as.integer}.}
\item{epsilon}{Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{0.1}.}
\item{optimistic}{Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{1}.}
}
\description{
Part of the mtdrl_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{agent}, as built in the body.
}
\keyword{internal}
\examples{
mtdrl_TabularHistoryAgent(n_arms = 5L)
}
