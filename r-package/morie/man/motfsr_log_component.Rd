% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{motfsr_log_component}
\alias{motfsr_log_component}
\title{motfsr_log_component}
\usage{
motfsr_log_component(theta, coded, i, j, w, comp) 
}
\arguments{
\item{theta}{A vector; indexed elementwise.}
\item{coded}{A vector; indexed elementwise.}
\item{i}{See Usage.}
\item{j}{Numeric; combined arithmetically in the body.}
\item{w}{A count; the body uses it as \code{seq_len(...)}.}
\item{comp}{Passed to \code{==}.}
}
\description{
Part of the motfsr_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{tot}, as built in the body.
}
\references{
  Bailey & Elkan (1994) ISMB-94, 28-36.
  Bailey & Elkan (1995) ISMB-95, 21-29.
Base R only.
}
\keyword{internal}
