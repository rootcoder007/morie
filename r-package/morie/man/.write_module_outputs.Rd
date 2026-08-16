% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.write_module_outputs}
\alias{.write_module_outputs}
\title{.write_module_outputs}
\usage{
.write_module_outputs(outputs, output_dir = NULL)
}
\arguments{
\item{outputs}{A vector; indexed elementwise.}
\item{output_dir}{Optional; may be \code{NULL}. Passed to \code{is.null}.}
}
\description{
Part of the modules implementation; see the file header for the source it follows.
}
\value{
The value of \code{outputs}, as built in the body.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.write_module_outputs(V)
}
