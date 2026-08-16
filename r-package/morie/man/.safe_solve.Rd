% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.safe_solve}
\alias{.safe_solve}
\title{Solve / pseudo-inverse helper}
\usage{
.safe_solve(A)
}
\arguments{
\item{A}{A matrix; passed to \code{solve}.}
}
\description{
A step of the diagnostics implementation. Called by \code{compute_influence}, \code{compute_residuals}, \code{compute_vif} and 2 others in the module. See the file header for the source the module follows.
}
\value{
One of two values, depending on the branch taken.
}
\keyword{internal}
