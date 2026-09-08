% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{whole_part_subpart}
\alias{whole_part_subpart}
\title{whole_part_subpart}
\usage{
whole_part_subpart(masks, target_hierarchy = NULL) 
}
\arguments{
\item{masks}{A vector; its length is taken.}
\item{target_hierarchy}{Accepted by the signature and not used anywhere in the body.}
}
\description{
Part of the sammkr_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{assignment}, \code{sizes}, \code{nested}, \code{note}.
}
\keyword{internal}
\examples{
S <- c("a", "b", "c")
whole_part_subpart(S)
}
