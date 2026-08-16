% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.reinfc_advance_baseline}
\alias{.reinfc_advance_baseline}
\title{.reinfc_advance_baseline}
\usage{
.reinfc_advance_baseline(state, baseline, gamma, r)
}
\arguments{
\item{state}{A list; the body reads \code{$v1}, \code{$v2} from it.}
\item{baseline}{One of \code{"comparison"}, \code{"mean"}.}
\item{gamma}{Numeric; combined arithmetically in the body.}
\item{r}{Numeric; combined arithmetically in the body.}
}
\description{
Part of the reinfc_native implementation; see the file header for the source it follows.
}
\value{
One of two values, depending on the branch taken.
}
\keyword{internal}
