% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_gru4r_gru}
\alias{morie_gru4r_gru}
\title{One GRU update (single layer)}
\usage{
morie_gru4r_gru(x, h, Wz, Uz, Wr, Ur, Wh, Uh)
}
\arguments{
\item{x}{Input vector (length d).}
\item{h}{Hidden state (length n).}
\item{Wz,Uz}{Update-gate linear maps.}
\item{Wr,Ur}{Reset-gate linear maps.}
\item{Wh,Uh}{Candidate-h linear maps.}
}
\description{
Update gate, reset gate, candidate, and the convex combination \code{(1 - z) h + z hh} that defines a GRU.
}
\value{
New hidden state.
}
\keyword{internal}
