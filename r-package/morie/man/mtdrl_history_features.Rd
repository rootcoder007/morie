% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{mtdrl_history_features}
\alias{mtdrl_history_features}
\title{mtdrl_history_features}
\usage{
mtdrl_history_features(history, n_arms) 
}
\arguments{
\item{history}{A vector; its length is taken and its elements indexed.}
\item{n_arms}{Numeric; combined arithmetically in the body.}
}
\description{
Part of the mtdrl_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{feat}, as built in the body.
}
\keyword{internal}
\examples{
mtdrl_history_features(history = data.frame(x = c(1, 2, 3, 4), y = c(2, 4, 5, 9)), n_arms = 5L)
}
