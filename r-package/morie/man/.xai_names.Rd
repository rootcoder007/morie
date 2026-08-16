% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.xai_names}
\alias{.xai_names}
\title{.xai_names}
\usage{
.xai_names(feature_names, d)
}
\arguments{
\item{feature_names}{Optional; may be \code{NULL}. Coerced to character by the body, with \code{as.character}.}
\item{d}{A count; the body uses it as \code{seq_len(...)}.}
}
\description{
A step of the fairness_xai implementation. Called by \code{morie_fairness_xai_ale}, \code{morie_fairness_xai_ceteris_paribus}, \code{morie_fairness_xai_partial_dependence} and 2 others in the module. See the file header for the source the module follows.
}
\value{
The value of \code{nm}, as built in the body.
}
\keyword{internal}
\examples{
.xai_names(feature_names = c("a", "b", "c"), d = 3L)
}
