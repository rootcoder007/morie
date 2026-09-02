% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{composite_distance}
\alias{composite_distance}
\title{composite_distance}
\usage{
composite_distance(tnf_a, tnf_b, cov_a = NULL, cov_b = NULL, len_a = NULL, len_b = NULL, 
    w_abundance = 0.5) 
}
\arguments{
\item{tnf_a}{Coerced to numeric by the body, with \code{as.numeric}.}
\item{tnf_b}{Coerced to numeric by the body, with \code{as.numeric}.}
\item{cov_a}{Optional; may be \code{NULL}. A vector; its length is taken.}
\item{cov_b}{Optional; may be \code{NULL}. Passed to \code{is.null}.}
\item{len_a}{Optional; may be \code{NULL}. Passed to \code{is.null}.}
\item{len_b}{Optional; may be \code{NULL}. Passed to \code{is.null}.}
\item{w_abundance}{Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{0.5}.}
}
\description{
Part of the metabd_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{distance}, \code{composition}, \code{abundance}, \code{abundance_usable}, \code{confidence}, \code{effective_weight}, \code{note}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
composite_distance(V, V)
}
