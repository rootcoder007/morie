% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.copy_legacy_artifacts}
\alias{.copy_legacy_artifacts}
\title{.copy_legacy_artifacts}
\usage{
.copy_legacy_artifacts(relative_paths, output_dir, root = file.path(.legacy_reference_root(), "six", "outputs"))
}
\arguments{
\item{relative_paths}{See Usage.}
\item{output_dir}{Passed to \code{file.path}.}
\item{root}{Passed to \code{file.path}.}
}
\description{
Part of the study_reporting implementation; see the file header for the source it follows.
}
\value{
The value of \code{copied}, as built in the body.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.copy_legacy_artifacts(V, V)
}
