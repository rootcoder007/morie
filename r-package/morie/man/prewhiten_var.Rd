% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{prewhiten_var}
\alias{prewhiten_var}
\title{VAR prewhitening, equation 2.2}
\usage{
prewhiten_var(v, order = 1L, cap = .EIGENVALUE_CAP, adjust = TRUE) 
}
\description{
VAR prewhitening, equation 2.2
}
\arguments{
\item{v}{Argument \code{v}; see Usage.}
\item{order}{Argument \code{order}; see Usage.}
\item{cap}{Argument \code{cap}; see Usage.}
\item{adjust}{Argument \code{adjust}; see Usage.}
}
\value{
A list with \code{a_list}, \code{resid}, \code{D}.
}
\keyword{internal}
