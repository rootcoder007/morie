% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.tbl_footnotes_render}
\alias{.tbl_footnotes_render}
\title{.tbl_footnotes_render}
\usage{
.tbl_footnotes_render(reg, fmt = "text")
}
\arguments{
\item{reg}{A list; the body reads \code{$notes} from it.}
\item{fmt}{Passed to \code{switch}. Defaults to \code{"text"}.}
}
\description{
Part of the tables_pub implementation; see the file header for the source it follows.
}
\value{
A character value.
}
\keyword{internal}
\examples{
D <- data.frame(x = c(1, 2, 3, 4), y = c(2, 4, 5, 9))
.tbl_footnotes_render(D)
}
