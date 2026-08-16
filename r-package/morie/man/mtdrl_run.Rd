% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{mtdrl_run}
\alias{mtdrl_run}
\title{mtdrl_run}
\usage{
mtdrl_run(tasks, agent, episode_length = 100, n_arms = NULL, seed = 0, reset_between_episodes = TRUE)
}
\arguments{
\item{tasks}{See Usage.}
\item{agent}{See Usage.}
\item{episode_length}{Defaults to \code{100}.}
\item{n_arms}{Defaults to \code{NULL}.}
\item{seed}{Defaults to \code{0}.}
\item{reset_between_episodes}{Defaults to \code{TRUE}.}
}
\description{
Part of the mtdrl_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{estimate}, \code{mean_reward}, \code{total_reward}, \code{regret}, \code{reward_by_step}, \code{optimal_action_rate}, \code{episode_reward}, \code{n_episodes}, \code{episode_length}, \code{n_arms}, \code{method}.
}
\keyword{internal}
