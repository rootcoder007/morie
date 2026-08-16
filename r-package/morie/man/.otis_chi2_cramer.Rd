% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.otis_chi2_cramer}
\alias{.otis_chi2_cramer}
\title{.otis_chi2_cramer}
\usage{
.otis_chi2_cramer(tbl)
}
\arguments{
\item{tbl}{A matrix; passed to \code{dim}.}
}
\description{
Part of the otis_all_analyze implementation; see the file header for the source it follows.
}
\value{
A list with \code{chi2}, \code{dof}, \code{pvalue}, \code{cramer_v}, \code{n}, \code{min_cell}, \code{min_expected}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.otis_chi2_cramer(V)
}
