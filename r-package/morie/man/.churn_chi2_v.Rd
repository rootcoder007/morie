% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.churn_chi2_v}
\alias{.churn_chi2_v}
\title{Chi-square + Cramer\\'s V on a 2x2-or-larger crosstab. Returns list}
\usage{
.churn_chi2_v(tbl, min_cell = 5L)
}
\arguments{
\item{tbl}{A matrix; passed to \code{dim}.}
\item{min_cell}{Defaults to \code{5L}.}
}
\description{
(chi2, p, v) with NA entries when the table is too sparse.
}
\value{
A list with \code{chi2}, \code{p}, \code{v}.
}
\keyword{internal}
