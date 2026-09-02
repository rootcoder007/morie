% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{motfsr_theta_from_subsequence}
\alias{motfsr_theta_from_subsequence}
\title{motfsr_theta_from_subsequence}
\usage{
motfsr_theta_from_subsequence(coded, i, j, w, L, mu, weight) 
}
\arguments{
\item{coded}{A vector; indexed elementwise.}
\item{i}{See Usage.}
\item{j}{Numeric; combined arithmetically in the body.}
\item{w}{A count; the body uses it as \code{seq_len(...)}.}
\item{L}{A count; the body uses it as \code{rep(...)}.}
\item{mu}{Coerced to numeric by the body, with \code{as.numeric}.}
\item{weight}{Numeric; combined arithmetically in the body.}
}
\description{
Part of the motfsr_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{theta}, as built in the body.
}
\references{
  Bailey & Elkan (1994) ISMB-94, 28-36.
  Bailey & Elkan (1995) ISMB-95, 21-29.
Base R only.
}
\keyword{internal}
