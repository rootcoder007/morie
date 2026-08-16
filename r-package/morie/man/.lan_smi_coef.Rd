% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.lan_smi_coef}
\alias{.lan_smi_coef}
\title{.lan_smi_coef}
\usage{
.lan_smi_coef(name, estimate, std_error)
}
\arguments{
\item{name}{Carried through into a list the body builds.}
\item{estimate}{Coerced to numeric by the body, with \code{as.numeric}.}
\item{std_error}{Coerced to numeric by the body, with \code{as.numeric}.}
}
\description{
Part of the laniyonu_smi_force_disparity implementation; see the file header for the source it follows.
}
\value{
A list with \code{name}, \code{estimate}, \code{std_error}.
}
\keyword{internal}
\examples{
.lan_smi_coef(name = c(1, 2, 3, 4, 5, 6, 7, 8), estimate = c(1, 2, 3, 4, 5, 6, 7, 8), std_error = c(1, 2, 3, 4, 5, 6, 7, 8))
}
