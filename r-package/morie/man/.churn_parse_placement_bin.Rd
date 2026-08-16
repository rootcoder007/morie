% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.churn_parse_placement_bin}
\alias{.churn_parse_placement_bin}
\title{Parse OTIS b09 placement-count bin labels:}
\usage{
.churn_parse_placement_bin(label)
}
\arguments{
\item{label}{Coerced to character by the body, with \code{as.character}.}
}
\description{
"1 placement" -> 1 "2 placements" -> 2 "6 to 10 placements" -> 8 (midpoint) "Greater than 40" -> 50 (boundary + 10)
}
\value{
One of two values, depending on the branch taken.
}
\keyword{internal}
\examples{
.churn_parse_placement_bin(label = 5L)
}
