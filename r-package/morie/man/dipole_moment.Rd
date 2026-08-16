% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{dipole_moment}
\alias{dipole_moment}
\title{Mu = sum_i q_i (r_i - r_c). A tensorial (here vector) property,}
\usage{
dipole_moment(charges, R, centre = NULL)
}
\arguments{
\item{charges}{Coerced to numeric by the body, with \code{as.numeric}.}
\item{R}{A matrix; passed to \code{as.matrix}.}
\item{centre}{Optional; may be \code{NULL}. Coerced to numeric by the body, with \code{as.numeric}.}
}
\description{
read off directly from equivariant atomwise quantities.
}
\value{
A list with \code{dipole}, \code{magnitude}, \code{note}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
dipole_moment(V, V)
}
