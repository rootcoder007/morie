% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{mtdrl_bandit_tasks}
\alias{mtdrl_bandit_tasks}
\title{mtdrl_bandit_tasks}
\usage{
mtdrl_bandit_tasks(n_arms = 2, n_tasks = 100, seed = 0, structure = "independent") 
}
\arguments{
\item{n_arms}{Coerced to integer by the body, with \code{as.integer}. Defaults to \code{2}.}
\item{n_tasks}{Coerced to integer by the body, with \code{as.integer}. Defaults to \code{100}.}
\item{seed}{Passed to \code{set.seed}. Defaults to \code{0}.}
\item{structure}{One of \code{"independent"}, \code{"paired"}. Defaults to \code{"independent"}.}
}
\description{
Part of the mtdrl_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{tasks}, as built in the body.
}
\keyword{internal}
\examples{
mtdrl_bandit_tasks()
}
