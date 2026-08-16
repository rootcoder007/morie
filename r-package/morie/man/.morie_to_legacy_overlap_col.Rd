% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_to_legacy_overlap_col}
\alias{.morie_to_legacy_overlap_col}
\title{Backwards-compatibility: the original equivalency joiners (added in}
\usage{
.morie_to_legacy_overlap_col(cw)
}
\arguments{
\item{cw}{A list; the body reads \code{$area_overlap_pct}, \code{$pct_140_in_158} from it.}
}
\description{
the same phase) read `area_overlap_pct`; remap to the new `pct_140_in_158` so they keep working.
}
\value{
The value of \code{cw}, as built in the body.
}
\keyword{internal}
\examples{
D <- data.frame(x = c(1, 2, 3, 4), y = c(2, 4, 5, 9))
.morie_to_legacy_overlap_col(D)
}
