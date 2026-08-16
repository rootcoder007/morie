% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_explor}
\alias{morie_explor}
\title{morie_explor}
\usage{
morie_explor(states, actions, next_states, n_actions = NULL, n_features = 8L, eta = 1, beta = 0.2, lr = 0.05, epochs = 1L, features = "inverse", discrete = TRUE, seed = 0L)
}
\arguments{
\item{states}{See Usage.}
\item{actions}{See Usage.}
\item{next_states}{See Usage.}
\item{n_actions}{Defaults to \code{NULL}.}
\item{n_features}{Defaults to \code{8L}.}
\item{eta}{Defaults to \code{1}.}
\item{beta}{Defaults to \code{0.2}.}
\item{lr}{Defaults to \code{0.05}.}
\item{epochs}{Defaults to \code{1L}.}
\item{features}{Defaults to \code{"inverse"}.}
\item{discrete}{Defaults to \code{TRUE}.}
\item{seed}{Defaults to \code{0L}.}
}
\description{
Part of the explor_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{explor}.
}
\references{
Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017)
"Curiosity-driven Exploration by Self-supervised Prediction", ICML,
arXiv:1705.05363. Eqs. 2-7.
}
\keyword{internal}
