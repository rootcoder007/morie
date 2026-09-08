% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_penalised_tmle}
\alias{morie_penalised_tmle}
\title{Penalised doubly robust TMLE}
\usage{
morie_penalised_tmle(y, D, X, penalty = 0.05, iters = 100) 
}
\arguments{
\item{y}{Numeric outcome in \code{[0,1]}.}
\item{D}{Numeric treatment vector.}
\item{X}{Numeric covariate matrix.}
\item{penalty}{Lasso penalty.}
\item{iters}{Maximum targeting iterations.}
}
\description{
Both the propensity \code{g} and the outcome regression \code{bar Q} are fitted by post-lasso; the one-dimensional targeting step is unpenalised.
}
\value{
A list with the estimate, SE, CI, EIC, and support.
}
\keyword{internal}
