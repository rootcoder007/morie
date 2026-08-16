% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.dbforward}
\alias{.dbforward}
\title{.dbforward}
\usage{
.dbforward(x, h, g, level)
}
\arguments{
\item{x}{See Usage.}
\item{h}{Passed to \code{.dbstep}.}
\item{g}{Passed to \code{.dbstep}.}
\item{level}{A count; the body uses it as \code{seq_len(...)}.}
}
\description{
Part of the wave implementation; see the file header for the source it follows.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
