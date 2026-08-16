% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_gru4r}
\alias{morie_gru4r}
\title{Session-parallel mini-batches}
\usage{
morie_gru4r(sessions, batch_size)
}
\arguments{
\item{sessions}{List of integer vectors (length >= 2 each).}
\item{batch_size}{Number of parallel slots, between 1 and the   number of sessions.}
}
\description{
Hidasi et al. (2016) Sec. 3.1. Slot \code{b} holds one session; when that session runs out the next free session takes the slot and the slot is flagged for a hidden-state reset, because sessions are assumed independent.
}
\value{
A list with \code{steps} (one element per batch tick:   \code{input}, \code{target}, \code{reset}), \code{n_steps},   \code{batch_size}, \code{n_sessions}, \code{note}.
}
\keyword{internal}
