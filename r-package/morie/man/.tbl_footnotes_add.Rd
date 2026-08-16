% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.tbl_footnotes_add}
\alias{.tbl_footnotes_add}
\title{.tbl_footnotes_add}
\usage{
.tbl_footnotes_add(reg, text)
}
\arguments{
\item{reg}{A list; the body reads \code{$notes} from it.}
\item{text}{See Usage.}
}
\description{
Part of the tables_pub implementation; see the file header for the source it follows.
}
\value{
The value of \code{[}.
}
\keyword{internal}
\examples{
.tbl_footnotes_add(reg = data.frame(x = c(1, 2, 3, 4), y = c(2, 4, 5, 9)), text = 5L)
}
