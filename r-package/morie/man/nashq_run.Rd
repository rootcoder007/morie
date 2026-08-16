% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{nashq_run}
\alias{nashq_run}
\title{nashq_run}
\usage{
nashq_run(states, actions, step, rewards, gamma = 0.9, alpha = 0.5, epsilon = 0.1, episodes = 500, horizon = 50, start = NULL, selection = "global_optimal", terminal = NULL, seed = 0, agent = 0, tol = 1e-09)
}
\arguments{
\item{states}{See Usage.}
\item{actions}{A vector; its length is taken and its elements indexed.}
\item{step}{See Usage.}
\item{rewards}{See Usage.}
\item{gamma}{Numeric; combined arithmetically in the body. Defaults to \code{0.9}.}
\item{alpha}{Numeric; combined arithmetically in the body. Defaults to \code{0.5}.}
\item{epsilon}{Defaults to \code{0.1}.}
\item{episodes}{Defaults to \code{500}.}
\item{horizon}{Defaults to \code{50}.}
\item{start}{Defaults to \code{NULL}.}
\item{selection}{Defaults to \code{"global_optimal"}.}
\item{terminal}{Defaults to \code{NULL}.}
\item{seed}{Defaults to \code{0}.}
\item{agent}{Defaults to \code{0}.}
\item{tol}{Defaults to \code{1e-09}.}
}
\description{
Part of the nashq_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{estimate}, \code{q}, \code{policy}, \code{nash_values}, \code{stage_game_types}, \code{returns}, \code{mean_return_last}, \code{selection}, \code{method}.
}
\keyword{internal}
