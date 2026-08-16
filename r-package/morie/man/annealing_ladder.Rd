% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{annealing_ladder}
\alias{annealing_ladder}
\title{annealing_ladder}
\usage{
annealing_ladder(n_steps, phi_max = 50, phi_min = 0.1, kind = "geometric")
}
\arguments{
\item{n_steps}{Coerced to integer by the body, with \code{as.integer}.}
\item{phi_max}{Numeric; combined arithmetically in the body. Defaults to \code{50}.}
\item{phi_min}{Numeric; combined arithmetically in the body. Defaults to \code{0.1}.}
\item{kind}{One of \code{"geometric"}, \code{"linear"}. Defaults to \code{"geometric"}.}
}
\description{
Part of the smcopt_native implementation; see the file header for the source it follows.
}
\value{
Nothing; this branch always raises.
}
\keyword{internal}
\examples{
annealing_ladder(n_steps = 5L)
}
