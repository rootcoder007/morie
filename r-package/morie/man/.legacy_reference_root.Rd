% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.legacy_reference_root}
\alias{.legacy_reference_root}
\title{The legacy migration tree exists only in a source checkout; an}
\usage{
.legacy_reference_root()
}
\description{
installed package has no project root. Degrade to NA so callers (.copy_legacy_artifacts) simply copy nothing rather than erroring.
}
\value{
The value of \code{file.path}.
}
\keyword{internal}
\examples{
.legacy_reference_root()
}
