% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{soft_value_iteration}
\alias{soft_value_iteration}
\title{Soft value iteration on a deterministic tabular MDP}
\usage{
soft_value_iteration(states, actions, step, reward, gamma = 0.9, iters = 2000L, tol = 1e-14) 
}
\description{
The maximum-entropy \eqn{V^*} of eq. Q(s, a) = r(s) + gamma V(s\'), V(s) = log sum_a exp Q(s, a). Provided so callers can validate AIRL\'s recovered reward independently of the fit.
}
\arguments{
\item{states}{Argument \code{states}; see Usage.}
\item{actions}{Argument \code{actions}; see Usage.}
\item{step}{Argument \code{step}; see Usage.}
\item{reward}{Argument \code{reward}; see Usage.}
\item{gamma}{Argument \code{gamma}; see Usage.}
\item{iters}{Argument \code{iters}; see Usage.}
\item{tol}{Argument \code{tol}; see Usage.}
}
\value{
A list with \code{V}, \code{pi}.
}
\keyword{internal}
